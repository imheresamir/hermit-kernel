#![allow(clippy::type_complexity)]

use alloc::boxed::Box;
use alloc::collections::{BTreeMap, VecDeque};
use alloc::rc::Rc;
use alloc::sync::Arc;
#[cfg(feature = "smp")]
use alloc::vec::Vec;
use core::cell::RefCell;
use core::ptr;
use core::sync::atomic::{AtomicI32, AtomicU32, Ordering};

use ahash::RandomState;
use crossbeam_utils::Backoff;
use hashbrown::{HashMap, hash_map};
use hermit_sync::*;
#[cfg(target_arch = "riscv64")]
use riscv::register::sstatus;
use timer_interrupts::TimerList;

use crate::arch::kernel;
use crate::arch::kernel::core_local::*;
use crate::arch::kernel::scheduler::TaskStacks;
#[cfg(target_arch = "riscv64")]
use crate::arch::kernel::switch::switch_to_task;
#[cfg(target_arch = "x86_64")]
use crate::arch::kernel::switch::{switch_to_fpu_owner, switch_to_task};
use crate::arch::kernel::{get_processor_count, interrupts};
use crate::errno::Errno;
use crate::fd::{Fd, RawFd};
use crate::io;
use crate::scheduler::task::*;

pub mod task;
pub mod timer_interrupts;

static NO_TASKS: AtomicU32 = AtomicU32::new(0);
/// Map between Core ID and per-core scheduler
#[cfg(feature = "smp")]
static SCHEDULER_INPUTS: SpinMutex<Vec<&InterruptTicketMutex<SchedulerInput>>> =
	SpinMutex::new(Vec::new());
/// Map between Task ID and Queue of waiting tasks
static WAITING_TASKS: InterruptTicketMutex<BTreeMap<TaskId, VecDeque<TaskHandle>>> =
	InterruptTicketMutex::new(BTreeMap::new());
/// Map between Task ID and TaskHandle
static TASKS: InterruptTicketMutex<BTreeMap<TaskId, TaskHandle>> =
	InterruptTicketMutex::new(BTreeMap::new());

/// Unique identifier for a core.
pub type CoreId = u32;

#[cfg(feature = "smp")]
pub(crate) struct SchedulerInput {
	/// Queue of new tasks
	new_tasks: VecDeque<NewTask>,
	/// Queue of task, which are wakeup by another core
	wakeup_tasks: VecDeque<TaskHandle>,
}

#[cfg(feature = "smp")]
impl SchedulerInput {
	pub fn new() -> Self {
		Self {
			new_tasks: VecDeque::new(),
			wakeup_tasks: VecDeque::new(),
		}
	}
}

#[cfg_attr(any(target_arch = "x86_64", target_arch = "aarch64"), repr(align(128)))]
#[cfg_attr(
	not(any(target_arch = "x86_64", target_arch = "aarch64")),
	repr(align(64))
)]
pub(crate) struct PerCoreScheduler {
	/// Core ID of this per-core scheduler
	#[cfg(feature = "smp")]
	core_id: CoreId,
	/// Task which is currently running
	current_task: Rc<RefCell<Task>>,
	/// Idle Task
	idle_task: Rc<RefCell<Task>>,
	/// Task that currently owns the FPU
	#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
	fpu_owner: Rc<RefCell<Task>>,
	/// Queue of tasks, which are ready
	ready_queue: PriorityTaskQueue,
	/// Queue of tasks, which are finished and can be released
	finished_tasks: VecDeque<Rc<RefCell<Task>>>,
	/// Queue of blocked tasks, sorted by wakeup time.
	blocked_tasks: BlockedTaskQueue,
	/// Queue of timer interrupts.
	pub timers: TimerList,
}

pub(crate) trait PerCoreSchedulerExt {
	/// Triggers the scheduler to reschedule the tasks.
	/// Interrupt flag will be cleared during the reschedule
	fn reschedule(self);

	/// Terminate the current task on the current core.
	fn exit(self, exit_code: i32) -> !;
}

impl PerCoreSchedulerExt for &mut PerCoreScheduler {
	#[cfg(target_arch = "x86_64")]
	fn reschedule(self) {
		without_interrupts(|| {
			let Some(last_stack_pointer) = self.scheduler(true) else {
				return;
			};

			let (new_stack_pointer, is_idle) = {
				let borrowed = self.current_task.borrow();
				(
					borrowed.last_stack_pointer,
					borrowed.status == TaskStatus::Idle,
				)
			};

			if is_idle || Rc::ptr_eq(&self.current_task, &self.fpu_owner) {
				unsafe {
					switch_to_fpu_owner(last_stack_pointer, new_stack_pointer.as_u64() as usize);
				}
			} else {
				unsafe {
					switch_to_task(last_stack_pointer, new_stack_pointer.as_u64() as usize);
				}
			}
		});
	}

	/// Trigger an interrupt to reschedule the system
	#[cfg(target_arch = "aarch64")]
	fn reschedule(self) {
		use aarch64_cpu::asm::barrier::{NSH, SY, dsb, isb};

		use crate::arch::kernel::interrupts::SGI_RESCHED;

		dsb(NSH);
		isb(SY);

		let intid: u64 = u64::from(SGI_RESCHED);
		#[cfg(feature = "smp")]
		let core_id = self.core_id;
		#[cfg(not(feature = "smp"))]
		let core_id = 0;

		let target_list: u64 = 1u64 << u64::from(core_id);
		let sgi_value: u64 = (intid << 24) | target_list;

		// SAFETY: ICC_SGI1R_EL1 triggers an SGI to specified cores.
		// We bypass GicCpuInterface::send_sgi() due to an ABI bug where
		// Result<(), GicError> uses a hidden return pointer (x8) that callers
		// pass as NULL, crashing on the post-MSR store.
		unsafe {
			core::arch::asm!(
				"msr ICC_SGI1R_EL1, {value:x}",
				value = in(reg) sgi_value,
				options(nostack),
			);
		}

		interrupts::enable();
	}

	#[cfg(target_arch = "riscv64")]
	fn reschedule(self) {
		without_interrupts(|| self.scheduler(true));
	}

	fn exit(self, exit_code: i32) -> ! {
		// DISCRIMINATOR (per-task-exception-slot-design.md R4-FU2): userspace
		// task-1 panic `slice index 1610613287 (0x60000207)` — SPSR-shaped,
		// KNOWN app code => kernel-induced corruption. `exit()` is the
		// CONVERGENCE of every abort path (sys_abort, scheduler::abort, direct
		// exit) and runs BEFORE reschedule() switches away from the faulting
		// task. Dump the current task's saved State (36 u64) and flag any slot
		// (esp. x-slots 5..35) == 0x60000207. READ-ONLY.
		{
			let lsp = self.current_task.borrow().last_stack_pointer.as_u64();
			let tid = self.current_task.borrow().id;
			error!(
				"[ABORT-DUMP] task={tid:?} frame_base={lsp:#x} exit_code={exit_code} (0x60000207 = SPSR-shaped panic index)"
			);
			if lsp != 0 {
				let slot = lsp as *const u64;
				let mut hit_any = false;
				let mut hit_x = false;
				for i in 0..36u64 {
					let v = unsafe { core::ptr::addr_of!(*slot.add(i as usize)).read_volatile() };
					let is_x = i >= 5 && i <= 35;
					if v == 0x60000207 {
						hit_any = true;
						if is_x {
							hit_x = true;
						}
						error!(
							"[ABORT-DUMP] slot[{i}] @+{:#x} = 0x60000207  <<< MATCH (is_x={is_x})",
							8 * i
						);
					} else if i == 2 {
						error!("[ABORT-DUMP] slot[{i}] @+{:#x} = {:#x}  (spsr)", 8 * i, v);
					} else if i == 1 {
						error!("[ABORT-DUMP] slot[{i}] @+{:#x} = {:#x}  (elr)", 8 * i, v);
					}
				}
				error!(
					"[ABORT-DUMP] kernel_leak? any={hit_any} x_slot={hit_x} (any slot == 0x60000207)"
				);
			}
		}
		without_interrupts(|| {
			// Get the current task.
			let mut current_task_borrowed = self.current_task.borrow_mut();
			assert_ne!(
				current_task_borrowed.status,
				TaskStatus::Idle,
				"Trying to terminate the idle task"
			);

			// Finish the task and reschedule.
			debug!(
				"Finishing task {} with exit code {}",
				current_task_borrowed.id, exit_code
			);
			current_task_borrowed.status = TaskStatus::Finished;
			NO_TASKS.fetch_sub(1, Ordering::SeqCst);

			let current_id = current_task_borrowed.id;
			drop(current_task_borrowed);

			// wakeup tasks, which are waiting for task with the identifier id
			if let Some(mut queue) = WAITING_TASKS.lock().remove(&current_id) {
				while let Some(task) = queue.pop_front() {
					self.custom_wakeup(task);
				}
			}

			TASKS.lock().remove(&current_id);
		});

		self.reschedule();
		unreachable!()
	}
}

struct NewTask {
	tid: TaskId,
	func: unsafe extern "C" fn(usize),
	arg: usize,
	prio: Priority,
	core_id: CoreId,
	stacks: TaskStacks,
	object_map: Arc<RwSpinLock<HashMap<RawFd, Arc<async_lock::RwLock<Fd>>, RandomState>>>,
}

impl From<NewTask> for Task {
	fn from(value: NewTask) -> Self {
		let NewTask {
			tid,
			func,
			arg,
			prio,
			core_id,
			stacks,
			object_map,
		} = value;
		let mut task = Self::new(tid, core_id, TaskStatus::Ready, prio, stacks, object_map);
		task.create_stack_frame(func, arg);
		task
	}
}

impl PerCoreScheduler {
	/// Spawn a new task.
	pub unsafe fn spawn(
		func: unsafe extern "C" fn(usize),
		arg: usize,
		prio: Priority,
		core_id: CoreId,
		stack_size: usize,
	) -> TaskId {
		// Create the new task.
		let tid = get_tid();
		let stacks = TaskStacks::new(stack_size);
		let new_task = NewTask {
			tid,
			func,
			arg,
			prio,
			core_id,
			stacks,
			object_map: core_scheduler().get_current_task_object_map(),
		};
		// Add it to the task lists.
		let wakeup = {
			#[cfg(feature = "smp")]
			let mut input_locked = get_scheduler_input(core_id).lock();
			WAITING_TASKS.lock().insert(tid, VecDeque::with_capacity(1));
			TASKS.lock().insert(
				tid,
				TaskHandle::new(
					tid,
					prio,
					#[cfg(feature = "smp")]
					core_id,
				),
			);
			NO_TASKS.fetch_add(1, Ordering::SeqCst);

			#[cfg(feature = "smp")]
			if core_id == core_scheduler().core_id {
				let task = Rc::new(RefCell::new(Task::from(new_task)));
				core_scheduler().ready_queue.push(task);
				false
			} else {
				input_locked.new_tasks.push_back(new_task);
				true
			}
			#[cfg(not(feature = "smp"))]
			if core_id == 0 {
				let task = Rc::new(RefCell::new(Task::from(new_task)));
				core_scheduler().ready_queue.push(task);
				false
			} else {
				panic!("Invalid core_id {core_id}!")
			}
		};

		debug!("Creating task {tid} with priority {prio} on core {core_id}");

		if wakeup {
			kernel::wakeup_core(core_id);
		}

		tid
	}

	#[cfg(feature = "newlib")]
	fn clone_impl(&self, func: extern "C" fn(usize), arg: usize) -> TaskId {
		static NEXT_CORE_ID: AtomicU32 = AtomicU32::new(1);

		// Get the Core ID of the next CPU.
		let core_id: CoreId = {
			// Increase the CPU number by 1.
			let id = NEXT_CORE_ID.fetch_add(1, Ordering::SeqCst);

			// Check for overflow.
			if id == get_processor_count() {
				NEXT_CORE_ID.store(0, Ordering::SeqCst);
				0
			} else {
				id
			}
		};

		// Get the current task.
		let current_task_borrowed = self.current_task.borrow();

		// Clone the current task.
		let tid = get_tid();
		let clone_task = NewTask {
			tid,
			func,
			arg,
			prio: current_task_borrowed.prio,
			core_id,
			stacks: TaskStacks::new(current_task_borrowed.stacks.get_user_stack_size()),
			object_map: current_task_borrowed.object_map.clone(),
		};

		// Add it to the task lists.
		let wakeup = {
			#[cfg(feature = "smp")]
			let mut input_locked = get_scheduler_input(core_id).lock();
			WAITING_TASKS.lock().insert(tid, VecDeque::with_capacity(1));
			TASKS.lock().insert(
				tid,
				TaskHandle::new(
					tid,
					current_task_borrowed.prio,
					#[cfg(feature = "smp")]
					core_id,
				),
			);
			NO_TASKS.fetch_add(1, Ordering::SeqCst);
			#[cfg(feature = "smp")]
			if core_id == core_scheduler().core_id {
				let clone_task = Rc::new(RefCell::new(Task::from(clone_task)));
				core_scheduler().ready_queue.push(clone_task);
				false
			} else {
				input_locked.new_tasks.push_back(clone_task);
				true
			}
			#[cfg(not(feature = "smp"))]
			if core_id == 0 {
				let clone_task = Rc::new(RefCell::new(Task::from(clone_task)));
				core_scheduler().ready_queue.push(clone_task);
				false
			} else {
				panic!("Invalid core_id {core_id}!");
			}
		};

		// Wake up the CPU
		if wakeup {
			kernel::wakeup_core(core_id);
		}

		tid
	}

	#[cfg(feature = "newlib")]
	pub fn clone(&self, func: extern "C" fn(usize), arg: usize) -> TaskId {
		without_interrupts(|| self.clone_impl(func, arg))
	}

	/// Returns `true` if a reschedule is required
	#[inline]
	#[cfg(all(any(target_arch = "x86_64", target_arch = "riscv64"), feature = "smp"))]
	pub fn is_scheduling(&self) -> bool {
		self.current_task.borrow().prio < self.ready_queue.get_highest_priority()
	}

	#[inline]
	pub fn handle_waiting_tasks(&mut self) {
		without_interrupts(|| {
			crate::executor::run();
			self.blocked_tasks
				.handle_waiting_tasks(&mut self.ready_queue);
		});
	}

	/// Minimal, bounded wake-move used on the exception stack (E) path (Part B:
	/// "defer, don't relocate"). Moves woken/blocked tasks into `ready_queue`
	/// WITHOUT draining the async executor (which must run off E, on the reactor
	/// idle loop). The executor drain happens in `PerCoreScheduler::run()`.
	#[inline]
	pub fn wake_pending_tasks(&mut self) {
		without_interrupts(|| {
			self.blocked_tasks
				.handle_waiting_tasks(&mut self.ready_queue);
		});
	}

	#[cfg(not(feature = "smp"))]
	pub fn custom_wakeup(&mut self, task: TaskHandle) {
		without_interrupts(|| {
			let task = self.blocked_tasks.custom_wakeup(task);
			self.ready_queue.push(task);
		});
	}

	#[cfg(feature = "smp")]
	pub fn custom_wakeup(&mut self, task: TaskHandle) {
		if task.get_core_id() == self.core_id {
			without_interrupts(|| {
				let task = self.blocked_tasks.custom_wakeup(task);
				self.ready_queue.push(task);
			});
		} else {
			get_scheduler_input(task.get_core_id())
				.lock()
				.wakeup_tasks
				.push_back(task);
			// Wake up the CPU
			kernel::wakeup_core(task.get_core_id());
		}
	}

	#[inline]
	pub fn block_current_task(&mut self, wakeup_time: Option<u64>) {
		without_interrupts(|| {
			self.blocked_tasks
				.add(self.current_task.clone(), wakeup_time);
		});
	}

	#[inline]
	pub fn get_current_task_handle(&self) -> TaskHandle {
		without_interrupts(|| {
			let current_task_borrowed = self.current_task.borrow();

			TaskHandle::new(
				current_task_borrowed.id,
				current_task_borrowed.prio,
				#[cfg(feature = "smp")]
				current_task_borrowed.core_id,
			)
		})
	}

	#[inline]
	pub fn get_current_task_id(&self) -> TaskId {
		without_interrupts(|| self.current_task.borrow().id)
	}

	/// EL1t diagnostic helper: the current task's kernel_stack_top
	/// (= base + size), i.e. the address SP_EL0 is set to on EL1t return.
	/// Used only by the data-abort fault dumper to confirm whether a fault
	/// at `kernel_stack_top` is an off-by-one stack-top write (Option D §11.7).
	pub fn get_current_task_kernel_stack_top(&self) -> u64 {
		without_interrupts(|| {
			let t = self.current_task.borrow();
			t.stacks.get_kernel_stack().as_u64() + t.stacks.get_kernel_stack_size() as u64
		})
	}

	/// [ALLOC-TRACE] Returns (kernel_stack_base, kernel_stack_top, task_id_raw)
	/// for the currently running task, without allocating. Used by the
	/// allocator instrumentation to detect heap/stack overlap.
	pub fn get_current_task_kstack_bounds(&self) -> (u64, u64, i32) {
		without_interrupts(|| {
			let t = self.current_task.borrow();
			let base = t.stacks.get_kernel_stack().as_u64();
			let top = base + t.stacks.get_kernel_stack_size() as u64;
			(base, top, t.id.into())
		})
	}

	#[inline]
	pub fn get_current_task_object_map(
		&self,
	) -> Arc<RwSpinLock<HashMap<RawFd, Arc<async_lock::RwLock<Fd>>, RandomState>>> {
		without_interrupts(|| self.current_task.borrow().object_map.clone())
	}

	/// Map a file descriptor to their IO interface and returns
	/// the shared reference
	#[inline]
	pub fn get_object(&self, fd: RawFd) -> io::Result<Arc<async_lock::RwLock<Fd>>> {
		without_interrupts(|| {
			let current_task = self.current_task.borrow();
			let object_map = current_task.object_map.read();
			object_map.get(&fd).cloned().ok_or(Errno::Badf)
		})
	}

	/// Creates a new map between file descriptor and their IO interface and
	/// clone the standard descriptors.
	#[cfg(feature = "common-os")]
	#[cfg_attr(not(target_arch = "x86_64"), expect(dead_code))]
	pub fn recreate_objmap(&self) -> io::Result<()> {
		let mut map = HashMap::<RawFd, Arc<async_lock::RwLock<Fd>>, RandomState>::with_hasher(
			RandomState::with_seeds(0, 0, 0, 0),
		);

		without_interrupts(|| {
			let mut current_task = self.current_task.borrow_mut();
			let object_map = current_task.object_map.read();

			// clone standard file descriptors
			for i in 0..3 {
				if let Some(obj) = object_map.get(&i) {
					map.insert(i, obj.clone());
				}
			}

			drop(object_map);
			current_task.object_map = Arc::new(RwSpinLock::new(map));
		});

		Ok(())
	}

	/// Insert a new IO interface and returns a file descriptor as
	/// identifier to this object
	pub fn insert_object(&self, obj: Arc<async_lock::RwLock<Fd>>) -> io::Result<RawFd> {
		without_interrupts(|| {
			let current_task = self.current_task.borrow();
			let mut object_map = current_task.object_map.write();

			let new_fd = || -> io::Result<RawFd> {
				let mut fd: RawFd = 0;
				loop {
					if !object_map.contains_key(&fd) {
						break Ok(fd);
					} else if fd == RawFd::MAX {
						break Err(Errno::Overflow);
					}

					fd = fd.saturating_add(1);
				}
			};

			let fd = new_fd()?;
			object_map.insert(fd, obj.clone());
			Ok(fd)
		})
	}

	/// Duplicate a IO interface and returns a new file descriptor as
	/// identifier to the new copy
	pub fn dup_object(&self, fd: RawFd) -> io::Result<RawFd> {
		without_interrupts(|| {
			let current_task = self.current_task.borrow();
			let mut object_map = current_task.object_map.write();

			let obj = (*(object_map.get(&fd).ok_or(Errno::Inval)?)).clone();

			let new_fd = || -> io::Result<RawFd> {
				let mut fd: RawFd = 0;
				loop {
					if !object_map.contains_key(&fd) {
						break Ok(fd);
					} else if fd == RawFd::MAX {
						break Err(Errno::Overflow);
					}

					fd = fd.saturating_add(1);
				}
			};

			let fd = new_fd()?;
			match object_map.entry(fd) {
				hash_map::Entry::Occupied(_occupied_entry) => Err(Errno::Mfile),
				hash_map::Entry::Vacant(vacant_entry) => {
					vacant_entry.insert(obj);
					Ok(fd)
				}
			}
		})
	}

	pub fn dup_object2(&self, fd1: RawFd, fd2: RawFd) -> io::Result<RawFd> {
		without_interrupts(|| {
			let current_task = self.current_task.borrow();
			let mut object_map = current_task.object_map.write();

			let obj = object_map.get(&fd1).cloned().ok_or(Errno::Badf)?;

			match object_map.entry(fd2) {
				hash_map::Entry::Occupied(_occupied_entry) => Err(Errno::Mfile),
				hash_map::Entry::Vacant(vacant_entry) => {
					vacant_entry.insert(obj);
					Ok(fd2)
				}
			}
		})
	}

	/// Remove a IO interface, which is named by the file descriptor
	pub fn remove_object(&self, fd: RawFd) -> io::Result<Arc<async_lock::RwLock<Fd>>> {
		without_interrupts(|| {
			let current_task = self.current_task.borrow();
			let mut object_map = current_task.object_map.write();

			object_map.remove(&fd).ok_or(Errno::Badf)
		})
	}

	#[inline]
	pub fn get_current_task_prio(&self) -> Priority {
		without_interrupts(|| self.current_task.borrow().prio)
	}

	/// Returns reference to prio_bitmap
	#[allow(dead_code)]
	#[inline]
	pub fn get_priority_bitmap(&self) -> &u64 {
		self.ready_queue.get_priority_bitmap()
	}

	#[cfg(target_arch = "x86_64")]
	pub fn set_current_kernel_stack(&self) {
		let current_task_borrowed = self.current_task.borrow();
		let tss = unsafe { &mut *CoreLocal::get().tss.get() };

		let rsp = current_task_borrowed.stacks.get_kernel_stack()
			+ current_task_borrowed.stacks.get_kernel_stack_size() as u64
			- TaskStacks::MARKER_SIZE as u64;
		tss.privilege_stack_table[0] = rsp.into();
		CoreLocal::get().kernel_stack.set(rsp.as_mut_ptr());
		let ist_start = current_task_borrowed.stacks.get_interrupt_stack()
			+ current_task_borrowed.stacks.get_interrupt_stack_size() as u64
			- TaskStacks::MARKER_SIZE as u64;
		tss.interrupt_stack_table[0] = ist_start.into();
	}

	pub fn set_current_task_priority(&mut self, prio: Priority) {
		without_interrupts(|| {
			trace!("Change priority of the current task");
			self.current_task.borrow_mut().prio = prio;
		});
	}

	pub fn set_priority(&mut self, id: TaskId, prio: Priority) -> Result<(), ()> {
		trace!("Change priority of task {id} to priority {prio}");

		without_interrupts(|| {
			let task = get_task_handle(id).ok_or(())?;
			#[cfg(feature = "smp")]
			let other_core = task.get_core_id() != self.core_id;
			#[cfg(not(feature = "smp"))]
			let other_core = false;

			if other_core {
				warn!("Have to change the priority on another core");
			} else if self.current_task.borrow().id == task.get_id() {
				self.current_task.borrow_mut().prio = prio;
			} else {
				self.ready_queue
					.set_priority(task, prio)
					.expect("Do not find valid task in ready queue");
			}

			Ok(())
		})
	}

	#[cfg(target_arch = "riscv64")]
	pub fn set_current_kernel_stack(&self) {
		let current_task_borrowed = self.current_task.borrow();

		let stack = (current_task_borrowed.stacks.get_kernel_stack()
			+ current_task_borrowed.stacks.get_kernel_stack_size() as u64
			- TaskStacks::MARKER_SIZE as u64)
			.as_u64();
		CoreLocal::get().kernel_stack.set(stack);
	}

	/// Save the FPU context for the current FPU owner and restore it for the current task,
	/// which wants to use the FPU now.
	#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
	pub fn fpu_switch(&mut self) {
		if !Rc::ptr_eq(&self.current_task, &self.fpu_owner) {
			debug!(
				"Switching FPU owner from task {} to {}",
				self.fpu_owner.borrow().id,
				self.current_task.borrow().id
			);

			self.fpu_owner.borrow_mut().last_fpu_state.save();
			self.current_task.borrow().last_fpu_state.restore();
			self.fpu_owner = self.current_task.clone();
		}
	}

	/// Check if a finished task could be deleted.
	fn cleanup_tasks(&mut self) {
		// Pop the first finished task and remove it from the TASKS list, which implicitly deallocates all associated memory.
		while let Some(finished_task) = self.finished_tasks.pop_front() {
			debug!("Cleaning up task {}", finished_task.borrow().id);
		}
	}

	#[cfg(feature = "smp")]
	pub fn check_input(&mut self) {
		let mut input_locked = CoreLocal::get().scheduler_input.lock();

		while let Some(task) = input_locked.wakeup_tasks.pop_front() {
			let task = self.blocked_tasks.custom_wakeup(task);
			self.ready_queue.push(task);
		}

		while let Some(new_task) = input_locked.new_tasks.pop_front() {
			let task = Rc::new(RefCell::new(Task::from(new_task)));
			self.ready_queue.push(task.clone());
		}
	}

	/// Only the idle task should call this function.
	/// Set the idle task to halt state if not another
	/// available.
	pub fn run() -> ! {
		let backoff = Backoff::new();

		loop {
			let core_scheduler = core_scheduler();
			interrupts::disable();

			// run async tasks
			crate::executor::run();

			// do housekeeping
			#[cfg(feature = "smp")]
			core_scheduler.check_input();
			core_scheduler.cleanup_tasks();

			if core_scheduler.ready_queue.is_empty() {
				if backoff.is_completed() {
					interrupts::enable_and_wait();
					backoff.reset();
				} else {
					interrupts::enable();
					backoff.snooze();
				}
			} else {
				interrupts::enable();
				core_scheduler.reschedule();
				backoff.reset();
			}
		}
	}

	#[inline]
	#[cfg(target_arch = "aarch64")]
	pub fn get_last_stack_pointer(&self) -> memory_addresses::VirtAddr {
		self.current_task.borrow().last_stack_pointer
	}

	/// Per-task exception slot design (per-task-exception-slot-design.md):
	/// ensure `task` has a scratch slot before it is dispatched. On first
	/// dispatch the frame lives on the kernel stack (slot == -1); we acquire a
	/// slot and copy the frame into it, updating `last_stack_pointer` to the
	/// slot frame base so the switch path (start.s) publishes the correct
	/// `scratch_slot`. If the pool is exhausted, evict a stale blocked task
	/// (claim-before-copy) and retry. Bounded recursion via SLOTS_PER_CORE.
	#[cfg(target_arch = "aarch64")]
	fn ensure_slot(&mut self, task: &Rc<RefCell<Task>>) {
		use crate::arch::kernel::slot_pool;

		// Already resident in a slot? Nothing to do (last_stack_pointer already
		// points at the slot frame base from a prior dispatch).
		{
			let b = task.borrow();
			if b.slot >= 0 && b.frame_location == FrameLocation::InSlot {
				return;
			}

			// EL1h GATE (R5 errata): slot relocation is ONLY valid for EL1t
			// frames. An EL1t context resumes on SP_EL0, so its State frame
			// can live anywhere — the frame's address carries no meaning to
			// the resumed code. An EL1h context, however, resumes on SP_EL1,
			// and trap_exit's 18 ldp pops define resume SP = frame_base + 288:
			// the frame's ADDRESS *is* the resume stack pointer. Copying an
			// EL1h frame into a slot makes the task resume with SP = slot top
			// on a 288-byte "stack" that immediately runs into the guard page
			// (PROVEN: idle suspended at EL1h, dispatched from slot 1,
			// faulted writing slot1_top+0x50 = guard). Dispatch EL1h frames
			// IN PLACE — they already sit on their own kernel stack, which is
			// exactly the stack they must resume on.
			// Frame word[0] = spsel saved by trap_entry (1 = EL1h).
			let frame = b.last_stack_pointer.as_u64();
			if frame != 0 {
				let spsel = unsafe { core::ptr::read_volatile(frame as *const u64) };
				if spsel & 1 == 1 {
					return;
				}
			}
		}

		// Fast path: pool has a free slot.
		{
			let mut b = task.borrow_mut();
			if slot_pool::dispatch_acquire_slot(&mut b) {
				return;
			}
		}

		// Slow path: pool exhausted. Evict a stale blocked resident, then retry.
		// Scan this core's blocked tasks for a victim (exclude the task being
		// resumed — no self-eviction, design INV-S4).
		let self_id = task.borrow().id;
		let mut victim: Option<Rc<RefCell<Task>>> = None;
		for blocked in self.blocked_tasks.iter() {
			let b = blocked.borrow();
			if b.id == self_id {
				continue;
			}
			if b.frame_location != FrameLocation::InSlot || b.slot < 0 {
				continue;
			}
			victim = Some(blocked.clone());
			break; // simplest selection: first eligible resident
		}
		if let Some(v) = victim {
			let slot_idx = v.borrow().slot as usize;
			slot_pool::evict_victim(&mut v.borrow_mut(), slot_idx);
		}

		// Retry acquisition. If still exhausted (e.g. no blocked victim),
		// fall back to running on the kernel-stack frame: the switch path
		// publishes scratch_slot = last_stack_pointer + 288 = kernel_stack_top,
		// so the frame lands on the task's OWN kernel stack — safe (no shared
		// E), just without per-slot isolation. This is graceful degradation,
		// not a crash (design §5.3 bounded exhaustion fallback).
		let mut b = task.borrow_mut();
		if !slot_pool::dispatch_acquire_slot(&mut b) {
			warn!(
				"ensure_slot: pool exhausted, task {} running on kernel-stack frame (no isolated slot)",
				self_id
			);
		}
	}

	/// Triggers the scheduler to reschedule the tasks.
	/// Interrupt flag must be cleared before calling this function.
	/// `drain_exec` = true drains the async executor (use OFF the exception
	/// stack, e.g. the reactor idle loop); false skips it (use on the exception
	/// stack E path — Part B defers executor work to the reactor).
	pub fn scheduler(&mut self, drain_exec: bool) -> Option<*mut usize> {
		// run background tasks (deferred off E by the IRQ path)
		if drain_exec {
			crate::executor::run();
		}

		// Someone wants to give up the CPU
		// => we have time to cleanup the system
		self.cleanup_tasks();

		// Get information about the current task.
		let (id, last_stack_pointer, prio, status) = {
			let mut borrowed = self.current_task.borrow_mut();
			(
				borrowed.id,
				ptr::from_mut(&mut borrowed.last_stack_pointer).cast::<usize>(),
				borrowed.prio,
				borrowed.status,
			)
		};

		let mut new_task = None;

		if status == TaskStatus::Running {
			// A task is currently running.
			// Check if a task with a equal or higher priority is available.
			if let Some(task) = self.ready_queue.pop_with_prio(prio) {
				new_task = Some(task);
			}
		} else {
			if status == TaskStatus::Finished {
				// Mark the finished task as invalid and add it to the finished tasks for a later cleanup.
				self.current_task.borrow_mut().status = TaskStatus::Invalid;
				self.finished_tasks.push_back(self.current_task.clone());
			}

			// No task is currently running.
			// Check if there is any available task and get the one with the highest priority.
			if let Some(task) = self.ready_queue.pop() {
				// This available task becomes the new task.
				debug!("Task is available.");
				new_task = Some(task);
			} else if status != TaskStatus::Idle {
				// The Idle task becomes the new task.
				debug!("Only Idle Task is available.");
				new_task = Some(self.idle_task.clone());
			}
		}

		let task = new_task?;
		// There is a new task we want to switch to.

		// Handle the current task.
		if status == TaskStatus::Running {
			// Mark the running task as ready again and add it back to the queue.
			self.current_task.borrow_mut().status = TaskStatus::Ready;
			self.ready_queue.push(self.current_task.clone());
		}

		// Handle the new task and get information about it.
		#[cfg(target_arch = "aarch64")]
		self.ensure_slot(&task);

		// §4D (invariant-assertion-surface-area.md): update CoreLocal.kernel_sp
		// to the new task's kernel-stack top BEFORE the asm switch path runs.
		// The asm switch (start.s) publishes scratch_slot (@24) but NOT kernel_sp
		// (@16). call_with_kernel_stack reads kernel_sp to set SP_EL1 for deep
		// handler work. If kernel_sp is stale (boot/E value), deep work runs on
		// the wrong stack → overflow → silent memory corruption.
		//
		// kernel_stack_top = get_kernel_stack() + get_kernel_stack_size(): the
		// top of the 128 KiB kernel-stack region. For EL1t tasks the frame sits
		// at the BOTTOM of this region (frame_base ≈ kstack + kstack_size - 288),
		// so kernel_stack_top is well ABOVE the frame and safe for deep work.
		#[cfg(target_arch = "aarch64")]
		{
			let kernel_stack_top = task.borrow().stacks.get_kernel_stack().as_u64()
				+ task.borrow().stacks.get_kernel_stack_size() as u64;
			CoreLocal::get().set_kernel_sp(kernel_stack_top);
		}

		let (new_id, new_stack_pointer) = {
			let mut borrowed = task.borrow_mut();
			if borrowed.status != TaskStatus::Idle {
				// Mark the new task as running.
				borrowed.status = TaskStatus::Running;
			}

			(borrowed.id, borrowed.last_stack_pointer)
		};

		if id == new_id {
			return None;
		}

		// Tell the scheduler about the new task.
		if new_stack_pointer.as_usize() == 0 {
			error!(
				"SCHEDULER: switching to task {} with ZERO last_stack_pointer! (from task {})",
				new_id, id
			);
		}
		debug!(
			"Switching task from {} to {} (stack {:#X} => {:p})",
			id,
			new_id,
			unsafe { *last_stack_pointer },
			new_stack_pointer
		);
		#[cfg(not(target_arch = "riscv64"))]
		{
			self.current_task = task;
		}

		// Finally return the context of the new task.
		#[cfg(not(target_arch = "riscv64"))]
		return Some(last_stack_pointer);

		#[cfg(target_arch = "riscv64")]
		{
			if sstatus::read().fs() == sstatus::FS::Dirty {
				self.current_task.borrow_mut().last_fpu_state.save();
			}
			task.borrow().last_fpu_state.restore();
			self.current_task = task;
			unsafe {
				switch_to_task(last_stack_pointer, new_stack_pointer.as_usize());
			}
			None
		}
	}
}

fn get_tid() -> TaskId {
	static TID_COUNTER: AtomicI32 = AtomicI32::new(0);
	let guard = TASKS.lock();

	loop {
		let id = TaskId::from(TID_COUNTER.fetch_add(1, Ordering::SeqCst));
		if !guard.contains_key(&id) {
			return id;
		}
	}
}

#[inline]
pub(crate) fn abort() -> ! {
	core_scheduler().exit(-1)
}

/// Add a per-core scheduler for the current core.
pub(crate) fn add_current_core() {
	// Create an idle task for this core.
	let core_id = core_id();
	let tid = get_tid();
	let idle_task = Rc::new(RefCell::new(Task::new_idle(tid, core_id)));

	// Add the ID -> Task mapping.
	WAITING_TASKS.lock().insert(tid, VecDeque::with_capacity(1));
	TASKS.lock().insert(
		tid,
		TaskHandle::new(
			tid,
			IDLE_PRIO,
			#[cfg(feature = "smp")]
			core_id,
		),
	);
	// Initialize a scheduler for this core.
	debug!("Initializing scheduler for core {core_id} with idle task {tid}");
	let boxed_scheduler = Box::new(PerCoreScheduler {
		#[cfg(feature = "smp")]
		core_id,
		current_task: idle_task.clone(),
		#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
		fpu_owner: idle_task.clone(),
		idle_task,
		ready_queue: PriorityTaskQueue::new(),
		finished_tasks: VecDeque::new(),
		blocked_tasks: BlockedTaskQueue::new(),
		timers: TimerList::new(),
	});

	let scheduler = Box::into_raw(boxed_scheduler);
	set_core_scheduler(scheduler);
	#[cfg(feature = "smp")]
	{
		SCHEDULER_INPUTS.lock().insert(
			core_id.try_into().unwrap(),
			&CoreLocal::get().scheduler_input,
		);
	}
}

#[inline]
#[cfg(feature = "smp")]
fn get_scheduler_input(core_id: CoreId) -> &'static InterruptTicketMutex<SchedulerInput> {
	SCHEDULER_INPUTS.lock()[usize::try_from(core_id).unwrap()]
}

pub unsafe fn spawn(
	func: unsafe extern "C" fn(usize),
	arg: usize,
	prio: Priority,
	stack_size: usize,
	selector: isize,
) -> TaskId {
	static CORE_COUNTER: AtomicU32 = AtomicU32::new(1);

	let core_id = if selector < 0 {
		// use Round Robin to schedule the cores
		CORE_COUNTER.fetch_add(1, Ordering::SeqCst) % get_processor_count()
	} else {
		selector as u32
	};

	unsafe { PerCoreScheduler::spawn(func, arg, prio, core_id, stack_size) }
}

#[allow(clippy::result_unit_err)]
pub fn join(id: TaskId) -> Result<(), ()> {
	let core_scheduler = core_scheduler();

	debug!(
		"Task {} is waiting for task {}",
		core_scheduler.get_current_task_id(),
		id
	);

	loop {
		let mut waiting_tasks_guard = WAITING_TASKS.lock();

		let Some(queue) = waiting_tasks_guard.get_mut(&id) else {
			return Ok(());
		};

		queue.push_back(core_scheduler.get_current_task_handle());
		core_scheduler.block_current_task(None);

		// Switch to the next task.
		drop(waiting_tasks_guard);
		core_scheduler.reschedule();
	}
}

pub fn shutdown(arg: i32) -> ! {
	crate::syscalls::shutdown(arg)
}

fn get_task_handle(id: TaskId) -> Option<TaskHandle> {
	TASKS.lock().get(&id).copied()
}

#[cfg(all(target_arch = "x86_64", feature = "common-os"))]
pub(crate) static BOOT_ROOT_PAGE_TABLE: OnceCell<usize> = OnceCell::new();

#[cfg(all(target_arch = "x86_64", feature = "common-os"))]
pub(crate) fn get_root_page_table() -> usize {
	let current_task_borrowed = core_scheduler().current_task.borrow_mut();
	current_task_borrowed.root_page_table
}
