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
use core::time::Duration;

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
use crate::scheduler::supervisor::EntryPointId;

pub mod supervisor;
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
				let (hit_any, hit_x) =
					crate::diagnostics::dump_frame_magic(lsp, "exit");
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

			// R7.1 (2026-07-25): release the slot back to the pool so the
			// exiting task does not permanently occupy it. Without this,
			// every task exit reduces pool capacity by 1 (with
			// SLOTS_PER_CORE=3, three exits exhaust the pool on a core and
			// all later tasks fall back to the kernel-stack frame). This is
			// a prerequisite for Phase 7 (kill+resume would otherwise leak
			// a slot per kill). release_slot asserts frame_location==InSlot
			// and slot>=0; only release when actually slot-resident (the
			// kernel-stack-fallback path is not InSlot and must skip this).
			if current_task_borrowed.frame_location == FrameLocation::InSlot {
				kernel::slot_pool::release_slot(&current_task_borrowed);
			}

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

		// Phase 8 supervisor hook (option-d §9 / R8.6 / R9.2 / R9.6 / R9.7):
		// AFTER the Fork C FD cleanup (inside reschedule -> cleanup_tasks) the
		// exiting task is reaped. BEFORE reschedule() dispatches the next task,
		// consult the per-entry-point restart policy table. If the policy
		// permits (and the BEAM-style MaxN window allows), respawn THIS entry
		// point on the SAME core (enqueued on this core's ready_queue;
		// reschedule() dispatches it within one scheduler cycle — invisible to
		// other cores).
		//
		// (review finding #10: `self.current_task` is still the just-finished
		// task here — `reschedule()` only swaps it below. Read the entry point
		// + original priority NOW, before that swap.)
		//
		// (review finding #13: this hook fires on EVERY task exit — normal
		// `sys_exit` AND fault-driven `abort()` — not just faults. That is
		// intentional: `RestartPolicy` is the contract ("restart on death"),
		// whatever the exit cause. Tasks default to `None`, so normal exits
		// are unaffected.)
		{
			let finished = self.current_task.borrow();
			let ep_id = finished.entry_point_id;
			let entry = finished.entry;
			let entry_arg = finished.entry_arg;
			let prio = finished.prio; // (finding #12) inherit the original priority
			drop(finished);
			if supervisor::should_restart(ep_id) {
				// (finding #11) The entry pointer comes from the dead task's
				// struct and the very fault we just handled may have corrupted
				// it. Reject the two obvious garbage values (null and the
				// sentinel `usize::MAX`), AND require the pointer to fall
				// within the kernel image range (review N2): a corrupted code
				// pointer is far more likely to be a plausible-looking address
				// pointing at a data page, heap, or unmapped memory than 0 or
				// ~0. Jumping there would be a secondary fault with no
				// diagnostics — fail closed instead. `kernel_start_address` /
				// `kernel_end_address` are the same bounds already asserted for
				// the exception stacks at core_local.rs:142-148.
				let entry_addr = entry as usize;
				let in_image = entry_addr
					>= crate::mm::kernel_start_address().as_u64() as usize
					&& entry_addr < crate::mm::kernel_end_address().as_u64() as usize;
				if entry_addr == 0 || entry_addr == usize::MAX || !in_image {
					warn!(
						"[SUPERVISOR] refusing respawn of {:?}: entry pointer \
						 corrupt ({entry_addr:#x}, in_image={}) — not restarting",
						ep_id, in_image
					);
				} else {
					debug!(
						"[SUPERVISOR] restarting entry-point {:?} (policy permitted)",
						ep_id
					);
					// Respawn on the same core. `spawn` clones the parent's
					// object_map (Arc) — the new task starts with shared FDs,
					// exactly like a fresh fork. `ensure_slot` assigns its slot.
					unsafe {
						PerCoreScheduler::spawn(
							entry,
							entry_arg,
							prio,
							core_id(),
							crate::config::DEFAULT_STACK_SIZE,
						);
					}
				}
			}
		}

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
		let mut task = Self::new(tid, core_id, TaskStatus::Ready, prio, stacks);
		// Inherit the parent task's fd map (object_map) rather than the
		// fresh one `Task::new` allocates — otherwise spawned tasks lose
		// their inherited file descriptors.
		task.object_map = object_map;
		task.create_stack_frame(func, arg);
		// Phase 8: retain the original entry point + arg and tag it with the
		// stable AppTask entry-point index so `exit()` can respawn it when the
		// policy (default None) permits. R9.7: `func` is a bare `extern "C" fn`
		// code pointer, trivially re-spawnable.
		task.entry = func;
		task.entry_arg = arg;
		task.entry_point_id = EntryPointId::AppTask;
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
			// Per-task fd table (D1): DEEP-COPY the parent's table so the child
			// owns its own outer Arc. See clone_current_task_object_map / F2.
			object_map: core_scheduler().clone_current_task_object_map(),
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
			// Per-task fd table (D1): DEEP-COPY, don't share the outer Arc (F2).
			// `current_task_borrowed` is already held, so build the copy inline
			// rather than calling clone_current_task_object_map (which would
			// re-borrow current_task).
			object_map: {
				let src = current_task_borrowed.object_map.read();
				let mut map =
					HashMap::<RawFd, Arc<async_lock::RwLock<Fd>>, RandomState>::with_hasher(
						RandomState::with_seeds(0, 0, 0, 0),
					);
				for (fd, obj) in src.iter() {
					map.insert(*fd, obj.clone());
				}
				debug_assert_eq!(
					map.len(),
					src.len(),
					"INV-2: clone_task fd-table deep copy dropped an fd"
				);
				let new_object_map = Arc::new(RwSpinLock::new(map));
				debug_assert_eq!(
					Arc::strong_count(&new_object_map),
					1,
					"INV-1: clone_task fd-table outer Arc must be unique"
				);
				new_object_map
			},
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
			// M8.4 INV-R8.1: the executor drain is the I/O core's (core 0)
			// reactor job. Compute cores must NOT drain here — that is
			// decoupled by M8.2 (migrate_waiting_tasks). Until M8.2 lands,
			// gate the drain to the I/O core so the reactor remains the
			// sole drainer.
			if core_id() == 0 {
				crate::executor::run();
			}
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
			let (task, completed) = self.blocked_tasks.custom_wakeup(task);
			// T6/R1.4: only push if the wake completed (status flipped to
			// Ready). A deferred wake (BeingEvicted) returns completed=false
			// and is scheduled later by the eviction completion.
			if completed {
				self.ready_queue.push(task);
			}
		});
	}

	#[cfg(feature = "smp")]
	pub fn custom_wakeup(&mut self, task: TaskHandle) {
		if task.get_core_id() == self.core_id {
			without_interrupts(|| {
				let (task, completed) = self.blocked_tasks.custom_wakeup(task);
				if completed {
					self.ready_queue.push(task);
				}
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

	/// NEW-1 (option-d-per-task-slot-rebased.md §10): the per-task exception
	/// slot design's frame_location of the current task. Used by do_sync/do_error
	/// to assert the frame landed on the task's own scratch slot (InSlot), not a
	/// shared/foreign stack.
	pub fn get_current_task_frame_location(&self) -> FrameLocation {
		without_interrupts(|| self.current_task.borrow().frame_location)
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

	/// Deep-copy the current task's fd table into a FRESH per-task table for a
	/// child (POSIX fork semantics; fd-ownership-and-task-teardown.md §2/D1).
	///
	/// The OUTER `Arc` is NEW (strong_count 1), so the child owns its table and
	/// `Task::drop` can reclaim it (fixes finding F2 — the old
	/// `object_map.clone()` shared the outer Arc, so a dead task's `Task::drop`
	/// only decremented it and never reached the inner map). Each VALUE
	/// (`Arc<Fd>`) is Arc-cloned, so the underlying open descriptions ARE shared
	/// with the parent and refcounted (I8 enforced structurally). Iterates the
	/// FULL table (contrast `recreate_objmap`, which copies only fds 0-2).
	pub fn clone_current_task_object_map(
		&self,
	) -> Arc<RwSpinLock<HashMap<RawFd, Arc<async_lock::RwLock<Fd>>, RandomState>>> {
		without_interrupts(|| {
			let current_task = self.current_task.borrow();
			let src = current_task.object_map.read();
			let mut map = HashMap::<RawFd, Arc<async_lock::RwLock<Fd>>, RandomState>::with_hasher(
				RandomState::with_seeds(0, 0, 0, 0),
			);
			for (fd, obj) in src.iter() {
				map.insert(*fd, obj.clone());
			}
			// INV-2 (fd-ownership-and-task-teardown.md §7): the deep copy must
			// preserve every fd VALUE handle of the source table.
			debug_assert_eq!(
				map.len(),
				src.len(),
				"INV-2: fd-table deep copy dropped an fd (copied {} of {})",
				map.len(),
				src.len()
			);
			let new_object_map = Arc::new(RwSpinLock::new(map));
			// INV-1 (fd-ownership-and-task-teardown.md §7): the spawned child's
			// fd-table OUTER Arc must be unique (per-task table, not shared).
			debug_assert_eq!(
				Arc::strong_count(&new_object_map),
				1,
				"INV-1: spawned task's fd-table outer Arc must be unique"
			);
			new_object_map
		})
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
	///
	/// Stage 3 (Fork C, option-d-per-task-slot-rebased.md §6.2 / R9.1 / R9.8;
	/// AMENDED by fd-ownership-and-task-teardown.md — see caveats below):
	/// Dropping the finished `Task` (via the `Rc<RefCell<Task>>` refcount
	/// hitting 0 here) drops its `object_map` OUTER `Arc`. That drops each
	/// entry's INNER `Arc<RwLock<Fd>>`. Because the inner `Arc` is the unit of
	/// ownership, inner refcount==1 (task-PRIVATE FD) drops the `Fd` and runs
	/// `Socket::drop`, which does NON-BLOCKING teardown: smoltcp `close()`
	/// (queues the FIN) + `executor::network::flush_nic()` to transmit it
	/// (tcp.rs / udp.rs). It does NOT call `block_on` — that model was removed
	/// (R9.1 re-entrancy); see fd-ownership-and-task-teardown.md §5 (D2/D3).
	/// inner refcount>1 (shared FD, dup'd or inherited from parent) is only
	/// decremented and survives until the LAST holder exits (I8).
	///
	/// IMPLEMENTED (fd-ownership-and-task-teardown.md, F1/F2/F5 — Stages A–C):
	/// (1) F1: the refcount only reaches 0 because `scheduler()` resets
	///     `fpu_owner` off the dying task (line ~1173); without that the task
	///     is pinned and this drop NEVER runs. Guarded by INV-3(a)/(b).
	/// (2) F2 (FIXED, Stage B): `spawn`/`clone_task` now DEEP-COPY the fd table
	///     (`clone_current_task_object_map`, line ~658), so each task owns its
	///     outer Arc (strong_count 1) and this drop DOES reach the inner map —
	///     task-private FDs are reclaimed. Guarded by INV-1/INV-2.
	/// (3) F5/D4: transmission of a queued FIN is guaranteed by the reap-path
	///     `flush_nic()` poke after this fn returns (D4, INV-6/INV-7); the
	///     `Socket::drop`-side poke is belt-and-suspenders.
	///
	/// R9.8 guard: `Socket::drop`'s teardown (`flush_nic()` → NIC `poll_common`)
	/// runs the network poll and can PANIC (corrupted NIC state). We set THIS
	/// core's per-core `abort_zone` around the drop so any such panic halts via
	/// I4 (processor::halt) instead of calling `scheduler::shutdown(1)` on a
	/// half-torn-down task. `cleanup_tasks` runs from `scheduler()`, which is
	/// called by `reschedule()` AFTER `exit()`'s `without_interrupts` block has
	/// ended and `current_task_borrowed` is dropped (mod.rs:245/257) — so the
	/// context is safe.
	/// Returns the number of tasks reaped (D4): the caller uses this to poke the
	/// reactor (`flush_nic()`) exactly once after a non-empty reap, guaranteeing
	/// a queued FIN is transmitted even if the system would otherwise go
	/// straight to idle (fd-ownership-and-task-teardown.md §5.4).
	fn cleanup_tasks(&mut self) -> usize {
		let cl = CoreLocal::get();
		let mut reaped: usize = 0;
		// INVARIANT (I8 / Fork C): only FINISHED tasks are reaped here. A
		// RUNNING task must never be in `finished_tasks` — reaping it would
		// drop the live current task's resources out from under it.
		// R9.8 guard: set THIS core's per-core `abort_zone` around the drop
		// loop with SeqCst. The matching load in the panic handler
		// (lib.rs:450) uses Relaxed — safe because the store is STRICTLY
		// sequenced-before any panic that could observe it (the panic happens
		// inside this loop, after the store(true) and before the store(false)),
		// so there is no concurrent writer to order against (review finding #5).
		// It is ALSO intentionally NOT cleared if the loop panics: a panic in
		// the abort zone must halt (I4), not fall through to a half-cleared
		// state and continue (findings #6 / #18). The store(false) below is
		// reached only on the happy path.
		// INV-3(b) (fd-ownership-and-task-teardown.md §7): before entering the
		// abort zone, verify no task queued for reaping is still pinned by
		// `fpu_owner` (finding F1). A pinned task would have `Task::drop`
		// suppressed and its sockets would leak. Checked HERE, outside the
		// abort zone, because a debug_assert panic inside the zone would halt
		// the core (§7 rule: no assertions on abort-zone paths). INV-3(a) at
		// the `Finished` branch is the primary guard; this is defense in depth.
		#[cfg(debug_assertions)]
		for finished_task in &self.finished_tasks {
			debug_assert!(
				!Rc::ptr_eq(&self.fpu_owner, finished_task),
				"INV-3: reaping task {} still pinned by fpu_owner",
				finished_task.borrow().id
			);
		}
		cl.abort_zone.store(true, Ordering::SeqCst);
		while let Some(finished_task) = self.finished_tasks.pop_front() {
			debug_assert!(
				matches!(
					finished_task.borrow().status,
					TaskStatus::Finished | TaskStatus::Invalid
				),
				"cleanup_tasks: reaping a non-finished task (status={:?})",
				finished_task.borrow().status
			);
			debug!(
				"Cleaning up task {} (Fork C: task-private FDs reclaimed via Task::drop; shared FDs survive)",
				finished_task.borrow().id
			);
			// `finished_task` dropped here -> Task::drop -> object_map outer Arc
			// drop -> inner-Arc==1 FDs close (I8 satisfied).
			drop(finished_task);
			reaped += 1;
		}
		cl.abort_zone.store(false, Ordering::SeqCst);
		reaped
	}

	#[cfg(feature = "smp")]
	pub fn check_input(&mut self) {
		let mut input_locked = CoreLocal::get().scheduler_input.lock();

		while let Some(task) = input_locked.wakeup_tasks.pop_front() {
			// T6/R1.4: gate the push on completion. A deferred cross-core
			// wake (the task was BeingEvicted on this core) returns
			// completed=false; the eviction completion schedules it later.
			let (task, completed) = self.blocked_tasks.custom_wakeup(task);
			if completed {
				self.ready_queue.push(task);
			}
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

			// M8.4 (reactor, irq-handling-architecture.md "Core-role model"):
			// ONLY the I/O core (core 0) drains the async executor. Compute
			// cores defer all executor work to core 0's reactor loop (the
			// "defer, don't relocate" contract) so tens of KB of async frames
			// never land on a compute core's path. On a single-core (non-SMP)
			// build core 0 is the only core, so this is behavior-preserving.
			// INV-R8.1: executor::run() is drained ONLY on the I/O core.
			if core_id() == 0 {
				// run async tasks
				crate::executor::run();
			}

			// Spike 2 (pmr-band): fire the per-band executor harness exactly
			// once, on the BSP / I/O core, from the idle loop (which runs AFTER
			// install_handlers has registered the SGI 12/13 handlers). The harness
			// pends SGI_RT_BRIDGE; when IRQs are re-enabled below the ISR fires
			// and wakes the COOP future. Guarded internally to run once.
			#[cfg(feature = "pmr-band")]
			interrupts::pmr_band_maybe_trigger();

			// Spike 3 (pmr-coop-net, INV-P6): fire the COOP-band executor
			// harness once, on the I/O core, from the idle loop. Also assert the
			// per-core EOI in-flight counter is 0 at idle — under EOImode=1 the
			// split priority-drop + deactivate must be balanced (every IAR1 ack
			// matched by exactly one EOIR1 + one DIR). A non-zero value means a
			// dropped/duplicate EOI would wedge the GIC.
			#[cfg(feature = "pmr-coop-net")]
			{
				debug_assert_eq!(
					CoreLocal::get().eoi_inflight(),
					0,
					"INV-P6: eoi_inflight != 0 at idle — EOImode=1 EOI/DIR pairing unbalanced"
				);
				interrupts::pmr_coop_net_maybe_trigger();
			}

			// Spike 4 (stackful-continuations.md §3): spawn the continuation
			// self-test harness once (on the I/O core) and drain any READY
			// continuation. Runs AFTER install_handlers has registered the
			// SGI_CONT_WAKE handler. The harness pends the SGI while masked;
			// when IRQs are re-enabled below the ISR fires and the next
			// drain_ready() resumes the cont → prints the PASS marker.
			// Guarded internally to run once (trigger) / when pending (drain).
			#[cfg(feature = "continuations")]
			{
				crate::arch::kernel::continuations::continuation_maybe_trigger();
				crate::arch::kernel::continuations::drain_ready();
				// O6.1/O6.3 (§10.3): if a continuation teardown happened, poke
				// the reactor once (INV-6/INV-7 re-hosting the cleanup_tasks
				// flush_nic poke that §7 deletes). Core 0 only (NIC/reactor).
				#[cfg(feature = "net")]
				if core_id() == 0
					&& crate::arch::kernel::continuations::continuation_reaped()
				{
					crate::executor::network::flush_nic();
				}
				// O6 (Spike 7a): verify teardown-wave completion.
				crate::arch::kernel::continuations::continuation_teardown_verify();
			}

			// do housekeeping
			#[cfg(feature = "smp")]
			core_scheduler.check_input();
			let reaped = core_scheduler.cleanup_tasks();
			// D4 (fd-ownership-and-task-teardown.md §5.4): poke the reactor after
			// a non-empty reap so a queued FIN is transmitted. Only meaningful on
			// the I/O core (core 0), where the executor/NIC live.
			#[cfg(feature = "net")]
			if reaped > 0 && core_id() == 0 {
				crate::executor::network::flush_nic();
			}
			#[cfg(not(feature = "net"))]
			let _ = reaped;

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
	/// dispatch the frame lives on the kernel stack (slot == None); we acquire a
	/// slot and copy the frame into it, updating `last_stack_pointer` to the
	/// slot frame base so the switch path (start.s) publishes the correct
	/// `scratch_slot`. If the pool is exhausted, evict a stale blocked task
	/// (claim-before-copy) and retry. Bounded recursion via SLOTS_PER_CORE.
	#[cfg(target_arch = "aarch64")]
	fn ensure_slot(&mut self, task: &Rc<RefCell<Task>>) {
		use crate::arch::kernel::slot_pool;
		use crate::config::SLOTS_PER_CORE;

		// T2: update the staleness signal on every dispatch attempt (one
		// write per context switch; zero asm changes — R1.3).
		{
			let mut b = task.borrow_mut();
			b.last_touch = Duration::from_micros(kernel::processor::get_timer_ticks());
		}

		// Step 1: already resident in a slot? (or EL1h in-place via spsel gate)
		{
			let b = task.borrow();
			if b.slot.is_some() && b.frame_location == FrameLocation::InSlot {
				return;
			}

			// EL1h GATE (R5 errata): slot relocation is ONLY valid for EL1t
			// frames. An EL1t context resumes on SP_EL0, so its State frame
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
				let spsel = unsafe { ptr::read_volatile(frame as *const u64) };
				if spsel & 1 == 1 {
					return;
				}
			} else {
				// DEFENSE-IN-DEPTH (review B5): a task with
				// `last_stack_pointer == 0` and `frame_location == InSlot`
				// would skip this EL1h/spsel check. The only way to reach
				// here with frame==0 is a task whose frame was never set
				// (which would be a separate bug), so we intentionally fall
				// through to "treat as EL1t, acquire a slot" — the SAFE
				// default (acquiring a slot can't corrupt a kernel stack the
				// way in-place EL1h dispatch of a zero frame would).
			}
		}

		// Step 2 (R2.1): Evicted check MUST precede dispatch_acquire_slot.
		// An Evicted task has frame_location==Evicted (not InSlot), so it
		// fell through step 1. Route it through resume_from_evicted — the ONLY
		// Evicted->InSlot path (INV-S8). If the pool is full, resume_from_evicted
		// returns false (leaves Evicted) and we fall through to eviction below.
		{
			let is_evicted = task.borrow().frame_location == FrameLocation::Evicted;
			if is_evicted {
				let resumed = {
					let mut b = task.borrow_mut();
					slot_pool::resume_from_evicted(&mut b)
				};
				if resumed {
					return; // resumed into a fresh slot
				}
				// pool full: leave Evicted, fall through to eviction below
			}
		}

		// Step 3: fast path — pool has a free slot (non-Evicted task).
		{
			let mut b = task.borrow_mut();
			if slot_pool::dispatch_acquire_slot(&mut b) {
				return;
			}
		}

		// Steps 4-5: bounded recursive eviction (§6.1, INV-S5). Each iteration
		// evicts the stalest eligible blocked resident and retries acquisition
		// for OUR task. Bound = SLOTS_PER_CORE-1 (pool is finite; no infinite
		// loop). The Evicted->InSlot transition for any woken victim is
		// completed here (R1.1 caller-side wake completion).
		let self_id = task.borrow().id;
		let mut evictions = 0usize;
		loop {
			if let Some(v) = self.blocked_tasks.select_eviction_victim(self_id) {
				let Some(slot_idx) = v.borrow().slot else {
					continue; // invariant: select_eviction_victim only returns InSlot+Some; defensive
				};
				// T4: evict_victim returns whether a wake was deferred.
				let woken = slot_pool::evict_victim(&mut v.borrow_mut(), slot_idx);
				if woken {
					// T5/R1.1: complete the deferred wake now (frame is Evicted).
					// mark_ready returns true (status Blocked->Ready); push it.
					let completed =
						BlockedTaskQueue::mark_ready(&v);
					if completed {
						self.ready_queue.push(v);
					}
				}
				// Retry acquisition for OUR task.
				{
					let mut b = task.borrow_mut();
					if slot_pool::dispatch_acquire_slot(&mut b) {
						return;
					}
				}
				evictions += 1;
				// INV-S5 (doc §11): debug_assert tripwire — the loop bound
				// (SLOTS_PER_CORE-1) is the authoritative guard; this catches a
				// future edit that wrongly grows the loop before the break.
				debug_assert!(
					evictions <= (SLOTS_PER_CORE - 1),
					"ensure_slot: eviction count {} exceeds bound SLOTS_PER_CORE-1={}",
					evictions,
					SLOTS_PER_CORE - 1
				);
				if evictions >= (SLOTS_PER_CORE - 1) {
					break;
				}
			} else {
				break; // no eligible victim -> graceful degradation
			}
		}

		// Hard exhaustion beyond bound: run on kernel-stack frame (graceful
		// degradation, design §5.3). The switch path publishes
		// scratch_slot = last_stack_pointer + 288 = kernel_stack_top, so the
		// frame lands on the task's OWN kernel stack — safe (no shared E),
		// just without per-slot isolation. Not a crash.
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
		// M8.4 INV-R8.1: only the I/O core (core 0) drains the executor;
		// on compute cores drop drain_exec to honor the reactor contract.
		if drain_exec && core_id() == 0 {
			debug_assert!(
				core_id() == 0,
				"scheduler(drain_exec=true) off the I/O core violates M8.4 INV-R8.1"
			);
			crate::executor::run();
		}

		// Someone wants to give up the CPU
		// => we have time to cleanup the system
		let reaped = self.cleanup_tasks();
		// D4 (fd-ownership-and-task-teardown.md §5.4): if we reaped a task, poke
		// the reactor ONCE so any FIN its `Socket::drop` queued is transmitted
		// even if the system goes straight to idle. Runs here — after
		// `cleanup_tasks` cleared the abort_zone and BEFORE the current_task
		// borrow below (R9.1 clean-context rule). INV-6/INV-7.
		if reaped > 0 && core_id() == 0 {
			debug_assert!(
				!CoreLocal::get().abort_zone.load(Ordering::Relaxed),
				"INV-7: D4 reactor poke must run outside the abort zone"
			);
			#[cfg(feature = "net")]
			crate::executor::network::flush_nic();
		}
		#[cfg(not(feature = "net"))]
		let _ = reaped;

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
				//
				// If the dying task is the current FPU owner, release that reference:
				// `fpu_owner` holds an `Rc` clone of the task (set in `switch_to_fpu_owner`
				// whenever a task first touches the FPU). That clone would otherwise keep
				// the dead task's `Rc` alive forever, so `Task::drop` (which reclaims the
				// task's `object_map` and closes its sockets — Fork C / option-d §6) would
				// never run and accepted TCP connections would leak with no FIN sent to the
				// peer. Reset it to `idle_task`, the natural FPU owner while no app task runs.
				if Rc::ptr_eq(&self.fpu_owner, &self.current_task) {
					self.fpu_owner = self.idle_task.clone();
				}
				// INV-3(a) (fd-ownership-and-task-teardown.md §7): after the reset,
				// `fpu_owner` must NOT alias the finishing task, or its `Rc` clone
				// would pin the task and `Task::drop` (fd reclamation) would never
				// run (finding F1). This is the guarantee behind D2.
				debug_assert!(
					!Rc::ptr_eq(&self.fpu_owner, &self.current_task),
					"INV-3: fpu_owner still aliases the just-finished task {}",
					self.current_task.borrow().id
				);
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
	// Fail-stop must actually STOP. Pre-scheduler (early boot) there is no
	// task to exit: core_scheduler()'s unwrap would panic, and the panic
	// path re-faults without a scheduler -> infinite panic loop that drowns
	// the original diagnostics (observed via the Phase 5 double-fault
	// injection harness). Degrade to a hard CPU halt instead.
	match try_core_scheduler() {
		Some(s) => s.exit(-1),
		None => loop {
			kernel::processor::halt();
		},
	}
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
