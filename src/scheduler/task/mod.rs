#![allow(clippy::type_complexity)]

#[cfg(not(feature = "common-os"))]
pub(crate) mod tls;

use alloc::collections::{LinkedList, VecDeque};
use alloc::vec::Vec;
use alloc::rc::Rc;
use alloc::sync::Arc;
use core::cell::{Cell, RefCell};
use core::num::NonZeroU64;
use core::sync::atomic::{AtomicUsize, Ordering};
use core::time::Duration;
use core::{cmp, fmt};

use ahash::RandomState;
use crossbeam_utils::CachePadded;
use hashbrown::HashMap;
use hermit_sync::{OnceCell, RwSpinLock};
use memory_addresses::VirtAddr;

#[cfg(not(feature = "common-os"))]
use self::tls::Tls;
use super::timer_interrupts::{Source, create_timer_abs};
use crate::arch::kernel::core_local::*;
use crate::arch::kernel::processor::{self, FPUState};
use crate::arch::kernel::scheduler::TaskStacks;
use crate::fd::{Fd, RawFd, stdio};
use crate::scheduler::CoreId;
use crate::scheduler::supervisor::EntryPointId;

// ---------------------------------------------------------------------------
// Bitmap pool: static bump allocator for cache-padded priority bitmaps.
//
// Each TaskHandlePriorityQueue needs a CachePadded<u64> for its priority
// bitmap (to prevent false sharing with the adjacent queues array). But
// CachePadded<u64> is #[repr(align(128))], and propagating that alignment
// through the struct chain (TaskHandlePriorityQueue → RecursiveMutexState →
// TicketMutex → RecursiveMutex) breaks the FFI contract with newlib:
// newlib's pthread_once_t is only 8-byte aligned.
//
// Solution: store CachePadded<u64> instances in a static pool and reference
// them by index. The struct holds just a usize (8-byte aligned).
// ---------------------------------------------------------------------------

/// Maximum concurrent mutex/semaphore instances that use pooled bitmaps.
const MAX_BITMAP_POOL: usize = 256;

#[derive(Copy, Clone)]
#[repr(C, align(128))]
struct BitmapSlot {
	bitmap: CachePadded<u64>,
}

static mut BITMAP_POOL: [BitmapSlot; MAX_BITMAP_POOL] = [BitmapSlot {
	bitmap: CachePadded::new(0),
}; MAX_BITMAP_POOL];
static NEXT_BITMAP_ID: AtomicUsize = AtomicUsize::new(0);

fn alloc_bitmap_id() -> usize {
	let id = NEXT_BITMAP_ID.fetch_add(1, Ordering::Relaxed);
	assert!(
		id < MAX_BITMAP_POOL,
		"exceeded MAX_BITMAP_POOL ({MAX_BITMAP_POOL})"
	);
	id
}

/// Returns a raw pointer to the cache-padded bitmap at the given pool index.
///
/// # Safety
///
/// `id` must be a value previously returned by `alloc_bitmap_id()`.
#[inline]
unsafe fn bitmap_ptr(id: usize) -> *mut CachePadded<u64> {
	unsafe { core::ptr::addr_of_mut!(BITMAP_POOL[id].bitmap) }
}

/// Returns the most significant bit.
///
/// # Examples
///
/// ```
/// assert_eq!(msb(0), None);
/// assert_eq!(msb(1), 0);
/// assert_eq!(msb(u64::MAX), 63);
/// ```
#[inline]
fn msb(n: u64) -> Option<u32> {
	NonZeroU64::new(n).map(|n| u64::BITS - 1 - n.leading_zeros())
}

/// The status of the task - used for scheduling
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) enum TaskStatus {
	Invalid,
	Ready,
	Running,
	Blocked,
	Finished,
	Idle,
}

/// Unique identifier for a task (i.e. `pid`).
#[derive(PartialEq, Eq, PartialOrd, Ord, Debug, Clone, Copy)]
pub struct TaskId(i32);

impl TaskId {
	pub const fn into(self) -> i32 {
		self.0
	}

	pub const fn from(x: i32) -> Self {
		TaskId(x)
	}
}

impl fmt::Display for TaskId {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(f, "{}", self.0)
	}
}

/// Priority of a task
#[derive(PartialEq, Eq, PartialOrd, Ord, Debug, Clone, Copy)]
pub struct Priority(u8);

impl Priority {
	pub const fn into(self) -> u8 {
		self.0
	}

	pub const fn from(x: u8) -> Self {
		Priority(x)
	}
}

impl fmt::Display for Priority {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(f, "{}", self.0)
	}
}

#[allow(dead_code)]
pub const HIGH_PRIO: Priority = Priority::from(3);
pub const NORMAL_PRIO: Priority = Priority::from(2);
#[allow(dead_code)]
pub const LOW_PRIO: Priority = Priority::from(1);
pub const IDLE_PRIO: Priority = Priority::from(0);

/// Maximum number of priorities
pub const NO_PRIORITIES: usize = 31;

#[derive(Copy, Clone, Debug)]
pub(crate) struct TaskHandle {
	id: TaskId,
	priority: Priority,
	#[cfg(feature = "smp")]
	core_id: CoreId,
}

impl TaskHandle {
	pub fn new(id: TaskId, priority: Priority, #[cfg(feature = "smp")] core_id: CoreId) -> Self {
		Self {
			id,
			priority,
			#[cfg(feature = "smp")]
			core_id,
		}
	}

	#[cfg(feature = "smp")]
	pub fn get_core_id(&self) -> CoreId {
		self.core_id
	}

	pub fn get_id(&self) -> TaskId {
		self.id
	}

	pub fn get_priority(&self) -> Priority {
		self.priority
	}
}

impl Ord for TaskHandle {
	fn cmp(&self, other: &Self) -> cmp::Ordering {
		self.id.cmp(&other.id)
	}
}

impl PartialOrd for TaskHandle {
	fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
		Some(self.cmp(other))
	}
}

impl PartialEq for TaskHandle {
	fn eq(&self, other: &Self) -> bool {
		self.id == other.id
	}
}

impl Eq for TaskHandle {}

/// Realize a priority queue for task handles.
///
/// The priority bitmap is stored in a static pool of `CachePadded<u64>`
/// instances (see `BITMAP_POOL` above) rather than inline.  This keeps the
/// struct at 8-byte alignment so that `RecursiveMutex` (which contains this
/// type) is FFI-compatible with newlib's `pthread_once_t` (also 8-byte
/// aligned).  The pool entries are `#[repr(align(128))]` so cache-line
/// isolation is preserved.
#[derive(Default)]
pub(crate) struct TaskHandlePriorityQueue {
	queues: [Option<VecDeque<TaskHandle>>; NO_PRIORITIES],
	prio_bitmap_id: Cell<usize>,
}

impl TaskHandlePriorityQueue {
	/// Creates an empty priority queue for tasks.
	///
	/// The bitmap pool slot is allocated lazily on first access.
	pub const fn new() -> Self {
		Self {
			queues: [const { None }; NO_PRIORITIES],
			prio_bitmap_id: Cell::new(0),
		}
	}

	/// Ensure a bitmap pool slot has been allocated for this queue.
	fn ensure_bitmap(&self) -> usize {
		let id = self.prio_bitmap_id.get();
		if id == 0 {
			let new_id = alloc_bitmap_id();
			self.prio_bitmap_id.set(new_id);
			new_id
		} else {
			id
		}
	}

	/// Shared reference to the cache-padded bitmap.
	#[inline]
	fn bitmap(&self) -> &CachePadded<u64> {
		let id = self.ensure_bitmap();
		unsafe { &*bitmap_ptr(id) }
	}

	/// Exclusive reference to the cache-padded bitmap.
	///
	/// # Safety
	///
	/// Caller must hold `&mut self` (exclusive access to this queue).
	#[inline]
	fn bitmap_mut(&mut self) -> &mut CachePadded<u64> {
		let id = self.ensure_bitmap();
		unsafe { &mut *bitmap_ptr(id) }
	}

	/// Checks if the queue is empty.
	pub fn is_empty(&self) -> bool {
		self.bitmap().into_inner() == 0
	}

	/// Checks if the given task is in the queue. Returns `true` if the task
	/// was found.
	pub fn contains(&self, task: TaskHandle) -> bool {
		matches!(self.queues[task.priority.into() as usize]
			.as_ref(), Some(queue) if queue.iter().any(|queued| queued.id == task.id))
	}

	/// Add a task handle by its priority to the queue
	pub fn push(&mut self, task: TaskHandle) {
		let i = task.priority.into() as usize;
		//assert!(i < NO_PRIORITIES, "Priority {} is too high", i);

		**self.bitmap_mut() |= (1 << i) as u64;
		if let Some(queue) = &mut self.queues[i] {
			queue.push_back(task);
		} else {
			let mut queue = VecDeque::new();
			queue.push_back(task);
			self.queues[i] = Some(queue);
		}
	}

	fn pop_from_queue(&mut self, queue_index: usize) -> Option<TaskHandle> {
		let queue = self.queues[queue_index].as_mut()?;

		let task = queue.pop_front();

		if queue.is_empty() {
			**self.bitmap_mut() &= !(1 << queue_index as u64);
		}

		task
	}

	/// Pop the task handle with the highest priority from the queue
	pub fn pop(&mut self) -> Option<TaskHandle> {
		let i = msb(self.bitmap().into_inner())?;

		self.pop_from_queue(i as usize)
	}

	/// Remove a specific task handle from the priority queue. Returns `true` if
	/// the handle was in the queue.
	pub fn remove(&mut self, task: TaskHandle) -> bool {
		let queue_index = task.priority.into() as usize;
		//assert!(queue_index < NO_PRIORITIES, "Priority {} is too high", queue_index);

		let mut success = false;
		if let Some(queue) = &mut self.queues[queue_index] {
			let mut i = 0;
			while i != queue.len() {
				if queue[i].id == task.id {
					queue.remove(i);
					success = true;
				} else {
					i += 1;
				}
			}

			if queue.is_empty() {
				**self.bitmap_mut() &= !(1 << queue_index as u64);
			}
		}

		success
	}
}

/// Realize a priority queue for tasks
pub(crate) struct PriorityTaskQueue {
	queues: [LinkedList<Rc<RefCell<Task>>>; NO_PRIORITIES],
	prio_bitmap: u64,
}

impl PriorityTaskQueue {
	/// Creates an empty priority queue for tasks
	pub const fn new() -> PriorityTaskQueue {
		const EMPTY_LIST: LinkedList<Rc<RefCell<Task>>> = LinkedList::new();
		PriorityTaskQueue {
			queues: [EMPTY_LIST; NO_PRIORITIES],
			prio_bitmap: 0,
		}
	}

	/// Add a task by its priority to the queue
	pub fn push(&mut self, task: Rc<RefCell<Task>>) {
		let i = task.borrow().prio.into() as usize;
		//assert!(i < NO_PRIORITIES, "Priority {} is too high", i);

		self.prio_bitmap |= (1 << i) as u64;
		let queue = &mut self.queues[i];
		queue.push_back(task);
	}

	fn pop_from_queue(&mut self, queue_index: usize) -> Option<Rc<RefCell<Task>>> {
		let task = self.queues[queue_index].pop_front();
		if self.queues[queue_index].is_empty() {
			self.prio_bitmap &= !(1 << queue_index as u64);
		}

		task
	}

	/// Remove the task at index from the queue and return that task,
	/// or None if the index is out of range or the list is empty.
	fn remove_from_queue(
		&mut self,
		task_index: usize,
		queue_index: usize,
	) -> Option<Rc<RefCell<Task>>> {
		//assert!(prio < NO_PRIORITIES, "Priority {} is too high", prio);

		let queue = &mut self.queues[queue_index];
		if queue.len() < task_index {
			return None;
		}

		// Calling remove is unstable: https://github.com/rust-lang/rust/issues/69210
		let mut split_list = queue.split_off(task_index);
		let element = split_list.pop_front();
		queue.append(&mut split_list);
		if queue.is_empty() {
			self.prio_bitmap &= !(1 << queue_index as u64);
		}
		element
	}

	/// Returns true if the queue is empty.
	pub fn is_empty(&self) -> bool {
		self.prio_bitmap == 0
	}

	/// Returns reference to prio_bitmap
	#[allow(dead_code)]
	#[inline]
	pub fn get_priority_bitmap(&self) -> &u64 {
		&self.prio_bitmap
	}

	/// Pop the task with the highest priority from the queue
	pub fn pop(&mut self) -> Option<Rc<RefCell<Task>>> {
		let i = msb(self.prio_bitmap)?;

		self.pop_from_queue(i as usize)
	}

	/// Pop the next task, which has a higher or the same priority as `prio`
	pub fn pop_with_prio(&mut self, prio: Priority) -> Option<Rc<RefCell<Task>>> {
		let i = msb(self.prio_bitmap)?;

		if i < u32::from(prio.into()) {
			return None;
		}

		self.pop_from_queue(i as usize)
	}

	/// Returns the highest priority of all available task
	#[cfg(all(any(target_arch = "x86_64", target_arch = "riscv64"), feature = "smp"))]
	pub fn get_highest_priority(&self) -> Priority {
		let Some(i) = msb(self.prio_bitmap) else {
			return IDLE_PRIO;
		};

		Priority::from(i.try_into().unwrap())
	}

	/// Change priority of specific task
	pub fn set_priority(&mut self, handle: TaskHandle, prio: Priority) -> Result<(), ()> {
		let old_priority = handle.get_priority().into() as usize;
		let index = self.queues[old_priority]
			.iter()
			.position(|current_task| current_task.borrow().id == handle.id)
			.ok_or(())?;

		let task = self.remove_from_queue(index, old_priority).ok_or(())?;
		task.borrow_mut().prio = prio;
		self.push(task);
		Ok(())
	}
}

/// A task control block, which identifies either a process or a thread
#[cfg_attr(any(target_arch = "x86_64", target_arch = "aarch64"), repr(align(128)))]
#[cfg_attr(
	not(any(target_arch = "x86_64", target_arch = "aarch64")),
	repr(align(64))
)]
/// Per-task exception slot design: tracks where a task's 288-byte State
/// frame lives. See per-task-exception-slot-design.md §4.1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FrameLocation {
	/// Frame is in the task's assigned scratch slot.
	InSlot,
	/// Eviction in progress (claim flag set); serializes with the wake path.
	BeingEvicted,
	/// Frame was copied to the persistent (kernel-stack) frame; slot freed.
	Evicted,
}

pub(crate) struct Task {
	/// The ID of this context
	pub id: TaskId,
	/// Status of a task, e.g. if the task is ready or blocked
	pub status: TaskStatus,
	/// Task priority,
	pub prio: Priority,
	/// Last stack pointer before a context switch to another task
	pub last_stack_pointer: VirtAddr,
	/// Last stack pointer on the user stack before jumping to kernel space
	pub user_stack_pointer: VirtAddr,
	/// Last FPU state before a context switch to another task using the FPU
	pub last_fpu_state: FPUState,
	/// ID of the core this task is running on
	pub core_id: CoreId,
	/// Stack of the task
	pub stacks: TaskStacks,
	/// Per-task exception slot design: where this task's 288-byte State frame
	/// currently lives. `IN_SLOT` = in the assigned scratch slot; `EVICTED` =
	/// copied to the persistent (kernel-stack) frame; `BEING_EVICTED` = a
	/// claim is held mid-copy (serializes with the wake path, design §4.4).
	pub frame_location: FrameLocation,
	/// Index of this task's assigned scratch slot within the current core's
	/// pool (`None` = none). Valid only while `frame_location == IN_SLOT`.
	pub slot: Option<usize>,
	/// Eviction/wake protocol (slot-eviction-wake-protocol.md §4.4 INV-S3):
	/// set by the wake path (`mark_ready`) when it observes `BeingEvicted`,
	/// signalling the eviction copy is in flight; the eviction completion
	/// (`evict_victim` in slot_pool.rs) reads it and completes the wake.
	/// Owned by the owning core (single-core ownership), no atomics needed.
	pub wake_pending: bool,
	/// Staleness signal for victim selection (§4.1): time of the last
	/// dispatch of this task, as a `Duration` since boot. Typed as
	/// `Duration` (R4.5 / 2026-07-25) so the unit is carried in the type —
	/// `get_timer_ticks()` returns MICROSECONDS, and a bare `u64` threshold
	/// invited a 1000x ms/us mismatch (the old `LAST_TOUCH_THRESHOLD_MS=10`
	/// was compared against µs ticks, so it skipped nothing). Default 0
	/// (Duration::ZERO, very old -> most evictable) until first dispatch.
	pub last_touch: Duration,
	/// Mapping between file descriptor and the referenced IO interface
	pub object_map: Arc<RwSpinLock<HashMap<RawFd, Arc<async_lock::RwLock<Fd>>, RandomState>>>,
	/// Phase 8 (supervisor / restart, option-d §9 / R8.6 / R9.2): the task's
	/// ORIGINAL entry point + argument, retained so `exit()` can respawn it
	/// when its `EntryPointId`'s policy permits (BEAM-style). Reusing the bare
	/// `extern "C" fn` code pointer (not a closure) — see R9.7, the current
	/// spawn ABI takes `func: extern "C" fn(usize)`, trivially "cloneable".
	/// These are `pub(crate)` (review finding #14): only the scheduler needs
	/// to read them for a respawn; external code has no business overwriting
	/// a running task's entry point.
	pub(crate) entry: unsafe extern "C" fn(usize),
	/// Original entry argument (passed back to a respawn).
	pub(crate) entry_arg: usize,
	/// Stable entry-point index this task was spawned from (NOT a fn pointer;
	/// PIE/rebase-safe — see supervisor.rs). Keys the per-entry-point restart
	/// policy + counter table.
	pub(crate) entry_point_id: EntryPointId,
	/// Task Thread-Local-Storage (TLS)
	#[cfg(not(feature = "common-os"))]
	pub tls: Option<Tls>,
	// Physical address of the 1st level page table
	#[cfg(all(target_arch = "x86_64", feature = "common-os"))]
	pub root_page_table: usize,
}

pub(crate) trait TaskFrame {
	/// Create the initial stack frame for a new task
	fn create_stack_frame(&mut self, func: unsafe extern "C" fn(usize), arg: usize);
}

/// Phase 8 placeholder entry point for tasks built via `Task::new` directly
/// (the idle task and the base of spawned tasks). Never actually invoked: the
/// idle task runs via its own `new_idle` frame, and spawned tasks have `entry`
/// overwritten by `into_task`. Present only so the `entry` field has a valid
/// non-null `extern "C" fn(usize)` default. If a restart ever dispatches this,
/// halting is the safe, loud failure.
extern "C" fn default_placeholder_entry(_arg: usize) {
	panic!("default_placeholder_entry invoked — a task with an unset entry was restarted");
}

impl Task {
	pub fn new(
		tid: TaskId,
		core_id: CoreId,
		task_status: TaskStatus,
		task_prio: Priority,
		stacks: TaskStacks,
		object_map: Arc<RwSpinLock<HashMap<RawFd, Arc<async_lock::RwLock<Fd>>, RandomState>>>,
	) -> Task {
		debug!("Creating new task {tid} on core {core_id}");

		Task {
			id: tid,
			status: task_status,
			prio: task_prio,
			last_stack_pointer: VirtAddr::zero(),
			user_stack_pointer: VirtAddr::zero(),
			last_fpu_state: FPUState::new(),
			core_id,
			stacks,
			frame_location: FrameLocation::InSlot,
			slot: None,
			wake_pending: false,
			last_touch: Duration::ZERO,
			object_map,
			// Phase 8 defaults: placeholder entry, Idle index, no restart.
			// `into_task` (spawn) overwrites these for real application tasks.
			// `Task::new` is used both for the idle task (which sets its own
			// entry via new_idle's frame) and as the base for spawned tasks;
			// this placeholder is never invoked unless a restart policy is set
			// on a task that was built via `new()` directly (none are).
			entry: default_placeholder_entry,
			entry_arg: 0,
			entry_point_id: EntryPointId::Idle,
			#[cfg(not(feature = "common-os"))]
			tls: None,
			#[cfg(all(target_arch = "x86_64", feature = "common-os"))]
			root_page_table: crate::arch::mm::create_new_root_page_table(),
		}
	}

	pub fn new_idle(tid: TaskId, core_id: CoreId) -> Task {
		debug!("Creating idle task {tid}");

		/// Idle-task resume entry (aarch64 / Option D EL1t model).
		///
		/// The idle/boot task has no `SP_EL0` body stack and never re-enters
		/// via `task_start` the way spawned tasks do (option-d doc §2.3 / D6).
		/// Its `State` is built once at creation (see the `create_stack_frame`
		/// call below) with `elr = task_start` and `x0 = idle_entry`, so when
		/// the scheduler switches to the idle task, `trap_exit` restores this
		/// frame and `task_start(idle_entry, 0)` runs `idle_entry` -> `run()`.
		/// `run()` is the scheduler loop: it drains the executor, and when the
		/// ready queue is empty it calls `enable_and_wait()` (wfi); the IRQ
		/// handler (`do_irq`) then switches to a ready task. On the next park
		/// the scheduler switches back to idle and `run()` resumes from the
		/// wfi — i.e. the idle task re-enters `run()` by RESUME, not by fresh
		/// call, so the boot stack is not consumed per idle cycle.
		#[cfg(target_arch = "aarch64")]
		unsafe extern "C" fn idle_entry(_arg: usize) -> () {
			crate::scheduler::PerCoreScheduler::run();
		}

		/// All cores use the same mapping between file descriptor and the referenced object
		static OBJECT_MAP: OnceCell<
			Arc<RwSpinLock<HashMap<RawFd, Arc<async_lock::RwLock<Fd>>, RandomState>>>,
		> = OnceCell::new();

		if core_id == 0 {
			OBJECT_MAP
				.set(Arc::new(RwSpinLock::new(HashMap::<
					RawFd,
					Arc<async_lock::RwLock<Fd>>,
					RandomState,
				>::with_hasher(
					RandomState::with_seeds(0, 0, 0, 0),
				))))
				// This function is called once per core and thus only once on core 0.
				// Thus, this is the only place where we set OBJECT_MAP.
				.unwrap_or_else(|_| unreachable!());
			let objmap = OBJECT_MAP.get().unwrap().clone();
			stdio::setup(&mut objmap.write());
		}

		#[cfg(not(feature = "common-os"))]
		let tls = if cfg!(feature = "instrument-mcount") {
			Tls::from_env().inspect(Tls::set_thread_ptr)
		} else {
			None
		};

		let mut idle_task = Task {
			id: tid,
			status: TaskStatus::Idle,
			prio: IDLE_PRIO,
			last_stack_pointer: VirtAddr::zero(),
			user_stack_pointer: VirtAddr::zero(),
			last_fpu_state: FPUState::new(),
			core_id,
			// Option D / D4 tail: the idle task needs a *real* resume State so
			// the scheduler can switch to it (`last_stack_pointer != 0`). The
			// original `Boot` stack places that frame in the unmapped top guard
			// page (poison on read -> "unhandled exception: 0"), so give the
			// idle task a proper allocated `Common` stack like a spawned task.
			// `create_stack_frame` then builds a valid frame on it (proven by
			// the init task), and `trap_exit` restores it correctly.
			#[cfg(target_arch = "aarch64")]
			stacks: TaskStacks::new(crate::config::DEFAULT_STACK_SIZE),
			#[cfg(not(target_arch = "aarch64"))]
			stacks: TaskStacks::from_boot_stacks(),
			// Idle participates in the scratch-slot pool like any other EL1t
			// task (per-task-exception-slot-design.md §8.2). It starts with
			// no slot; the first dispatch allocates one and sets
			// `last_stack_pointer` to the slot frame base.
			frame_location: FrameLocation::InSlot,
			slot: None,
			wake_pending: false,
			last_touch: Duration::ZERO,
			object_map: OBJECT_MAP.get().unwrap().clone(),
			// Phase 8: the idle task's entry is idle_entry; it is never
			// restarted (policy None, EntryPointId::Idle).
			entry: idle_entry,
			entry_arg: 0,
			entry_point_id: EntryPointId::Idle,
			#[cfg(not(feature = "common-os"))]
			tls,
			#[cfg(all(target_arch = "x86_64", feature = "common-os"))]
			root_page_table: *crate::scheduler::BOOT_ROOT_PAGE_TABLE.get().unwrap(),
		};

		// D4 tail / idle-frame (Option D): build a real resume State for the
		// idle/boot task so `last_stack_pointer != 0`. On aarch64 we use a
		// `Common` stack (allocated) so `create_stack_frame` places the frame
		// in mapped memory, with `elr = task_start`, `x0 = idle_entry` ->
		// `idle_entry` -> `run()` (the scheduler loop).
		#[cfg(target_arch = "aarch64")]
		idle_task.create_stack_frame(idle_entry, 0);

		idle_task
	}
}

/*impl Drop for Task {
	fn drop(&mut self) {
		debug!("Drop task {}", self.id);
	}
}*/

struct BlockedTask {
	task: Rc<RefCell<Task>>,
	wakeup_time: Option<u64>,
}

impl BlockedTask {
	pub fn new(task: Rc<RefCell<Task>>, wakeup_time: Option<u64>) -> Self {
		Self { task, wakeup_time }
	}
}

pub(crate) struct BlockedTaskQueue {
	list: LinkedList<BlockedTask>,
}

impl BlockedTaskQueue {
	pub const fn new() -> Self {
		Self {
			list: LinkedList::new(),
		}
	}

	// T6 (slot-eviction-wake-protocol.md §5.1, R1.4): returns `true` iff it
	// newly flipped the task to Ready (caller must push to ready_queue);
	// `false` if deferred (BeingEvicted) or already Ready (no push).
	pub(crate) fn mark_ready(task: &RefCell<Task>) -> bool {
		let mut borrowed = task.borrow_mut();
		// T9-V-RW1: wake-path discriminator (§8.2). Logs the frame_location
		// the wake observed and whether a deferral fired. Confirms INV-S3
		// reader behaviour. STRIP in T10.
		info!(
			"[V-RW1] mark_ready task {} core {} frame_location={:?} wake_pending(was)={}",
			borrowed.id, borrowed.core_id, borrowed.frame_location, borrowed.wake_pending
		);
		debug!(
			"Waking up task {} on core {}",
			borrowed.id, borrowed.core_id
		);

		assert!(
			borrowed.core_id == core_id(),
			"Try to wake up task {} on the wrong core {} != {}",
			borrowed.id,
			borrowed.core_id,
			core_id()
		);

		// R1.4: tolerate an already-Ready task (e.g. double wake: timer +
		// explicit close together, or a second Evicted-path wake). The
		// invariant we protect is "do not double-push to ready_queue" and
		// "do not wake a Running/Invalid task" — NOT "must be Blocked".
		assert!(
			borrowed.status == TaskStatus::Blocked
				|| borrowed.status == TaskStatus::Ready,
			"Trying to wake up task {} in unexpected status {:?}",
			borrowed.id,
			borrowed.status
		);
		if borrowed.status == TaskStatus::Ready {
			// Already readied (concurrent Evicted-path wake or double wake).
			// Do not re-push; the task is already scheduled.
			return false;
		}

		// INV-S3 reader: the wake path defers while an eviction copy is in
		// flight. We never touch the frame here; we only record the wake and
		// let `evict_victim` complete it (R1.1: caller-side completion).
		match borrowed.frame_location {
			FrameLocation::BeingEvicted => {
				borrowed.wake_pending = true;
				false // deferred; eviction completion will re-enter + flip Ready
			}
			FrameLocation::Evicted | FrameLocation::InSlot => {
				borrowed.status = TaskStatus::Ready;
				true // newly readied
			}
		}
	}

	/// Blocks the given task for `wakeup_time` ticks, or indefinitely if None is given.
	pub fn add(&mut self, task: Rc<RefCell<Task>>, wakeup_time: Option<u64>) {
		{
			// Set the task status to Blocked.
			let mut borrowed = task.borrow_mut();
			debug!("Blocking task {}", borrowed.id);

			assert_eq!(
				borrowed.status,
				TaskStatus::Running,
				"Trying to block task {} which is not running",
				borrowed.id
			);
			borrowed.status = TaskStatus::Blocked;
		}

		let new_node = BlockedTask::new(task, wakeup_time);

		// Shall the task automatically be woken up after a certain time?
		if let Some(wt) = wakeup_time {
			let mut cursor = self.list.cursor_front_mut();

			while let Some(node) = cursor.current() {
				let node_wakeup_time = node.wakeup_time;
				if node_wakeup_time.is_none() || wt < node_wakeup_time.unwrap() {
					cursor.insert_before(new_node);

					create_timer_abs(Source::Scheduler, wt);
					return;
				}

				cursor.move_next();
			}

			create_timer_abs(Source::Scheduler, wt);
		}

		self.list.push_back(new_node);
	}

	/// Iterate the blocked tasks resident in this queue (used by the
	/// per-task exception-slot eviction protocol to pick a stale victim,
	/// per-task-exception-slot-design.md §4.2).
	pub fn iter(&self) -> impl Iterator<Item = &Rc<RefCell<Task>>> {
		self.list.iter().map(|node| &node.task)
	}

	/// T3 (slot-eviction-wake-protocol.md §4.1, R1.2): select an eviction
	/// victim for the per-task slot pool when exhausted.
	///
	/// Candidate filter (all required):
	///   - not the task being resumed (`exclude_id`)         [INV-S4 no self]
	///   - `frame_location == InSlot && slot >= 0`           [only slot residents]
	///   - `status == TaskStatus::Blocked`                   [R1.2: never evict a
	///     task that is InSlot + Ready — its frame would move under a task
	///     about to run]
	///
	/// Staleness policy (prefer the WORST victim to evict):
	///   primary  key: negated `wakeup_time` (far-deadline / no-deadline first)
	///   secondary key: `last_touch` ascending (oldest-touch first)
	/// A candidate touched within `LAST_TOUCH_THRESHOLD_MS` is skipped (too
	/// fresh, about to wake) -> returns None -> graceful degradation.
	///
	/// Returns the chosen `Rc<RefCell<Task>>`, or None if no eligible
	/// victim (caller degrades to kernel-stack frame).
	pub(crate) fn select_eviction_victim(
		&self,
		exclude_id: TaskId,
	) -> Option<Rc<RefCell<Task>>> {
		// TUNABLE (L2): deliberate build-time knob, not a hardcoded literal.
		// R4.5 (2026-07-25): typed as Duration so the unit is enforced by the
		// compiler. get_timer_ticks() returns MICROSECONDS; 10ms of staleness
		// tolerance = from_millis(10). (The old LAST_TOUCH_THRESHOLD_MS=10 was
		// compared against µs ticks -> 1000x too small, effectively "skip
		// nothing".) A candidate touched within this window is too fresh
		// (about to wake) -> returns None -> graceful degradation.
		const LAST_TOUCH_THRESHOLD: Duration = Duration::from_millis(10);
		let now = Duration::from_micros(processor::get_timer_ticks());

		let mut candidates: Vec<(Rc<RefCell<Task>>, u64)> = Vec::new();
		for node in self.list.iter() {
			let t = &node.task;
			let b = t.borrow();
			// INV-S4 (doc §11: assert!, compiled in release — self-eviction
			// would deadlock resume, a real bug not a recoverable condition).
			assert!(
				b.id != exclude_id,
				"select_eviction_victim: self-eviction attempt for task {} (INV-S4)",
				b.id
			);
			if b.frame_location != FrameLocation::InSlot || b.slot.is_none() {
				continue; // not a slot resident
			}
			if b.status != TaskStatus::Blocked {
				continue; // R1.2: never evict a readied/running task
			}
			// Primary staleness key: far deadline (or no deadline) first.
			let wake_key: u64 = match node.wakeup_time {
				Some(w) => u64::MAX - w, // smaller key = further in the future
				None => 0,               // no deadline = most evictable
			};
			candidates.push((t.clone(), wake_key));
		}
		// Order by primary key, then by last_touch ascending (oldest first).
		candidates.sort_by(|a, b| {
			a.1.cmp(&b.1)
				.then_with(|| a.0.borrow().last_touch.cmp(&b.0.borrow().last_touch))
		});
		// Skip too-fresh candidates (about to wake) -> graceful degradation.
		let chosen = candidates
			.into_iter()
			.find(|(t, _)| now.saturating_sub(t.borrow().last_touch) >= LAST_TOUCH_THRESHOLD)
			.map(|(t, _)| t);
		// T9-V-RW3: victim-selection discriminator (§8.2). Confirms the
		// staleness policy (NOT first-fit) drove the choice. STRIP in T10.
		match &chosen {
			Some(t) => info!(
				"[V-RW3] select_eviction_victim chose task {} (staleness policy)",
				t.borrow().id
			),
			None => info!("[V-RW3] select_eviction_victim: no eligible victim (degrade)"),
		}
		chosen
	}

	/// Manually wake up a blocked task. Returns the woken task and whether
	/// the wake actually completed (status flipped to Ready). A `false`
	/// completion means the wake was deferred (the task was `BeingEvicted`);
	/// the caller must NOT push to the ready queue in that case — the
	/// eviction completion will re-enter `mark_ready` and schedule it.
	/// (slot-eviction-wake-protocol.md §5.1 R1.4.)
	pub fn custom_wakeup(&mut self, task: TaskHandle) -> (Rc<RefCell<Task>>, bool) {
		let mut first_task = true;
		let mut cursor = self.list.cursor_front_mut();

		// Loop through all blocked tasks to find it.
		while let Some(node) = cursor.current() {
			if node.task.borrow().id == task.get_id() {
				// Remove it from the list of blocked tasks.
				let task_ref = node.task.clone();
				cursor.remove_current();

				// If this is the first task, adjust the One-Shot Timer to fire at the
				// next task's wakeup time (if any).
				if first_task
					&& let Some(wakeup) = cursor
						.current()
						.map_or_else(|| None, |node| node.wakeup_time)
				{
					create_timer_abs(Source::Scheduler, wakeup);
				}

				// Wake it up (returns true iff newly readied).
				let completed = Self::mark_ready(&task_ref);

				return (task_ref, completed);
			}

			first_task = false;
			cursor.move_next();
		}

		unreachable!();
	}

	/// Wakes up all tasks whose wakeup time has elapsed.
	///
	/// Should be called by the One-Shot Timer interrupt handler when the wakeup time for
	/// at least one task has elapsed.
	pub fn handle_waiting_tasks(&mut self, ready_queue: &mut PriorityTaskQueue) {
		// Get the current time.
		let time = processor::get_timer_ticks();

		// Get the wakeup time of this task and check if we have reached the first task
		// that hasn't elapsed yet or waits indefinitely.
		// This iterator has to be consumed to actually remove the elements.
		let newly_ready_tasks = self.list.extract_if(|blocked_task| {
			blocked_task
				.wakeup_time
				.is_some_and(|wakeup_time| wakeup_time < time)
		});

		for task in newly_ready_tasks {
			// T6/R1.4: only push if the wake actually completed (status
			// flipped to Ready). A deferred/already-Ready wake returns
			// false and must not be pushed again here.
			if Self::mark_ready(&task.task) {
				ready_queue.push(task.task);
			}
		}

		let new_task_wakeup_time = self.list.front().and_then(|task| task.wakeup_time);

		if let Some(wakeup) = new_task_wakeup_time {
			create_timer_abs(Source::Scheduler, wakeup);
		}
	}
}
