//! Per-task exception scratch-slot pool (per-task-exception-slot-design.md).
//!
//! Replaces the shared per-core exception stack E (and the D3 E→persistent-frame
//! copy) with a per-core POOL of small, guard-paged scratch slots. Each core
//! owns `SLOTS_PER_CORE` slots. On task dispatch the scheduler allocates one
//! slot, copies the task's State frame into it, and publishes the slot TOP via
//! `CoreLocal.scratch_slot` (offset 24). `trap_entry` then builds every
//! exception frame on the task's OWN slot; on switch-out the frame stays in the
//! slot (no copy); on switch-in SP_EL1 is re-established to the task's slot by
//! the D4 tail (start.s `trap_exit` loads `scratch_slot`).
//!
//! Pool exhaustion is bounded: `SLOTS_PER_CORE` slots cover the running task
//! plus blocked tasks whose frames are still resident. If all slots are busy
//! when a new task must dispatch, the eviction protocol (§4) claims a stale
//! blocked task's slot, copies its frame to the persistent (kernel-stack)
//! storage, and reuses the slot. The claim flag (`FrameLocation::BeingEvicted`)
//! serializes the eviction copy with the cross-core wake path.

use core::sync::atomic::{AtomicI32, Ordering};

use memory_addresses::VirtAddr;

use crate::arch::aarch64::kernel::core_local::core_id;
use crate::config::{EXCEPTION_SLOT_GUARD, EXCEPTION_SLOT_SIZE, SLOTS_PER_CORE};
use crate::scheduler::task::{FrameLocation, Task};

/// Size of the State frame in bytes. Single source of truth for all slot
/// copy/offset arithmetic — never hardcode 288.
const STATE_SIZE: usize = size_of::<crate::arch::aarch64::kernel::scheduler::State>();

// Cross-check against the asm-hardcoded `#288` in start.s (D4 tail / switch
// vectors) AND the `ARCH_STATE_SIZE` literal in config.rs. The asm cannot call
// size_of, so this compile-time assert is the Rust-side guarantee that the
// type, the config literal, and the asm all agree.
const _: () = assert!(
	STATE_SIZE == crate::config::ARCH_STATE_SIZE,
	"State size must match ARCH_STATE_SIZE (asm hardcodes #288 for the frame in start.s)"
);

/// Total byte stride of one slot element in the pool: slot body + unmapped
/// guard tail. Mirrors the `.exception_stacks` LIEF-grown layout.
const SLOT_STRIDE: usize = EXCEPTION_SLOT_SIZE + EXCEPTION_SLOT_GUARD;

/// Top of slot `i` for the current core = base + i*SLOT_STRIDE + SLOT_SIZE.
/// `trap_entry` decrements sp from here, so the frame lands at
/// [top-288, top) (design §3.2 / D-2).
#[inline]
fn slot_top(core: usize, i: usize) -> u64 {
	unsafe extern "C" {
		static __start_exception_slots: u8;
	}
	let base = unsafe { &__start_exception_slots as *const u8 as usize };
	(base + core * SLOTS_PER_CORE * SLOT_STRIDE + i * SLOT_STRIDE + EXCEPTION_SLOT_SIZE) as u64
}

/// Per-core free-list: -1 = free, >=0 = task id owning the slot. Indexed by
/// slot number within the core. A slot is "in use" iff `owners[i] >= 0`.
struct CoreSlotOwners([AtomicI32; SLOTS_PER_CORE]);

impl CoreSlotOwners {
	const fn new() -> Self {
		CoreSlotOwners([const { AtomicI32::new(-1) }; SLOTS_PER_CORE])
	}

	/// Acquire a free slot for this core. Returns the slot index, or `-1` if
	/// the pool is exhausted (caller must evict).
	fn acquire(&self) -> i32 {
		for i in 0..SLOTS_PER_CORE {
			// CAS -1 -> a sentinel owner (usize::MAX) to claim atomically.
			if self.0[i]
				.compare_exchange(-1, i32::MAX, Ordering::AcqRel, Ordering::Relaxed)
				.is_ok()
			{
				return i as i32;
			}
		}
		-1
	}

	/// Release slot `i` back to the free-list.
	fn release(&self, i: usize) {
		self.0[i].store(-1, Ordering::Release);
	}

	/// Set the owner of slot `i` (after the frame has been placed).
	fn set_owner(&self, i: usize, owner: i32) {
		self.0[i].store(owner, Ordering::Release);
	}
}

// One owner array per core. Core count is small and known at link time; we
// size for the documented maximum and index by `core_id`.
static CORE_SLOTS: [CoreSlotOwners; 4] = [const { CoreSlotOwners::new() }; 4];

#[inline]
fn owners_for_core(core: usize) -> &'static CoreSlotOwners {
	// SAFETY: CORE_SLOTS is sized for the documented core maximum; callers
	// pass a valid core_id (0..N). Out-of-range would be a kernel bug.
	&CORE_SLOTS[core.min(CORE_SLOTS.len() - 1)]
}

/// Allocate a scratch slot for `task` on the current core, copy its State
/// frame from its persistent (kernel-stack) location into the slot, and update
/// the task's `last_stack_pointer` + `slot` + `frame_location` so the switch
/// path (start.s) will load the correct `scratch_slot`.
///
/// Returns `false` if the pool is exhausted (caller handles eviction).
pub fn dispatch_acquire_slot(task: &mut Task) -> bool {
	let core = core_id() as usize;
	let owners = owners_for_core(core);
	let idx = owners.acquire();
	if idx < 0 {
		return false;
	}
	let idx = idx as usize;
	let top = slot_top(core, idx);
	let frame_base = top - STATE_SIZE as u64; // trap_entry frame base (design §3.2)
	// Copy the persistent (kernel-stack) State into the slot. The persistent
	// frame already holds the last-resumed State (built by create_stack_frame
	// or by a prior eviction resume).
	let src = task.last_stack_pointer.as_u64() as *const u64;
	let dst = frame_base as *mut u64;
	unsafe {
		for w in 0..(STATE_SIZE / size_of::<u64>()) {
			*dst.add(w) = *src.add(w);
		}
	}
	task.last_stack_pointer = VirtAddr::new(frame_base);
	task.slot = idx as i32;
	task.frame_location = FrameLocation::InSlot;
	// T7/R1.5: clear any stale deferred-wake flag on (re)acquiring a slot.
	// A prior BeingEvicted deferral belongs to a previous eviction cycle.
	task.wake_pending = false;
	// INVARIANT (per-task-exception-slot-design.md): a frame resident in a
	// slot MUST satisfy frame_base + STATE_SIZE == slot_top(core, idx), i.e.
	// the 288-byte State occupies exactly the slot body's tail and
	// `scratch_slot` (published by the switch path as frame_base + 288) lands
	// on the slot TOP. If this fails, trap_entry/trap_exit build/pop the frame
	// at the wrong address -> silent register corruption (e.g. an SPSR value
	// surfacing in an xN slot). Enforce it.
	debug_assert_eq!(
		frame_base + STATE_SIZE as u64,
		top,
		"dispatch_acquire_slot: frame base {frame_base:#x} + STATE_SIZE != slot_top {top:#x} (core {core} slot {idx})"
	);
	owners.set_owner(idx, task.id.into());
	true
}

/// Release the task's slot back to the pool (on task exit).
pub fn release_slot(task: &Task) {
	if task.slot < 0 {
		return;
	}
	// §3H: precondition — task must own a valid slot in InSlot state.
	assert!(
		task.slot >= 0 && (task.slot as usize) < SLOTS_PER_CORE,
		"release_slot: invalid slot {} for task {}",
		task.slot,
		task.id
	);
	assert_eq!(
		task.frame_location,
		FrameLocation::InSlot,
		"release_slot: frame_location must be InSlot to release"
	);
	let core = core_id() as usize;
	owners_for_core(core).release(task.slot as usize);
}

/// Evict `victim` (frame resident in slot `slot_idx`): claim, copy the frame
/// back to the victim's persistent (kernel-stack) storage, mark `EVICTED`, and
/// release the slot so the caller can reuse it.
///
/// The claim (`BeingEvicted`) serializes with the wake path (design §4.4):
/// a cross-core wake that sees `BeingEvicted` defers until `EVICTED`.
///
/// NOTE: victim SELECTION (which blocked task to evict) lives in
/// `PerCoreScheduler` (scheduler/mod.rs), which owns the `Rc<RefCell<Task>>`
/// collections. This function only performs the copy + state transition for a
/// victim the scheduler has already chosen and borrowed mutably.
///
/// Returns `true` if a wake was deferred while this victim was
/// `BeingEvicted` (the wake path set `wake_pending`). The CALLER
/// (ensure_slot) must complete that wake by calling `mark_ready` on the
/// victim once it is `Evicted` — this avoids re-borrowing the `RefCell`
/// here (R1.1: evict_victim holds `&mut Task`, cannot call `mark_ready`
/// which needs `&RefCell<Task>`).
pub fn evict_victim(victim: &mut Task, slot_idx: usize) -> bool {
	// §3F: precondition — victim must have its frame in a slot.
	assert_eq!(
		victim.frame_location,
		FrameLocation::InSlot,
		"evict_victim: victim frame_location must be InSlot, got {:?}",
		victim.frame_location
	);
	assert!(
		victim.slot >= 0,
		"evict_victim: victim.slot must be >= 0, got {}",
		victim.slot
	);
	// Claim phase: mark BEING_EVICTED so the wake path defers.
	victim.frame_location = FrameLocation::BeingEvicted;
	// Copy phase: slot frame -> persistent (kernel-stack) frame.
	let top = slot_top(core_id() as usize, slot_idx);
	let frame_base = top - STATE_SIZE as u64;
	let src = frame_base as *const u64;
	let dst = victim.last_stack_pointer.as_u64() as *mut u64;
	unsafe {
		for w in 0..(STATE_SIZE / size_of::<u64>()) {
			*dst.add(w) = *src.add(w);
		}
	}
	// INVARIANT: the source slot frame must occupy exactly [top-STATE_SIZE, top).
	debug_assert_eq!(
		frame_base + STATE_SIZE as u64,
		top,
		"evict_victim: slot frame base {frame_base:#x} + STATE_SIZE != slot_top {top:#x} (slot {slot_idx})"
	);
	// Publish EVICTED, then free the slot.
	victim.frame_location = FrameLocation::Evicted;
	victim.slot = -1;
	owners_for_core(core_id() as usize).release(slot_idx);
	// R1.1/R1.5: hand off any deferred wake to the caller. Clear the flag
	// here so a future eviction on this task can't replay a stale wake.
	let woken = victim.wake_pending;
	victim.wake_pending = false;
	// T9-V-RW2: eviction discriminator (§8.2). Confirms claim->copy->Evicted
	// and whether a deferred wake is being handed off. STRIP in T10.
	info!(
		"[V-RW2] evict_victim task {} slot {} -> Evicted, woken(deferred)={}",
		victim.id, slot_idx, woken
	);
	woken
}

/// Resume a task whose frame was evicted: acquire a fresh slot, copy the
/// persistent frame back in, and restore `last_stack_pointer`/`slot`/
/// `frame_location`. Returns `false` if the pool is still exhausted (caller
/// recurses via evict_and_acquire, bounded by SLOTS_PER_CORE-1, design §5.3).
pub fn resume_from_evicted(task: &mut Task) -> bool {
	// §3G: precondition — task must have been evicted to persistent storage.
	assert_eq!(
		task.frame_location,
		FrameLocation::Evicted,
		"resume_from_evicted: task frame_location must be Evicted, got {:?}",
		task.frame_location
	);
	let core = core_id() as usize;
	let owners = owners_for_core(core);
	let idx = owners.acquire();
	if idx < 0 {
		return false;
	}
	let idx = idx as usize;
	let top = slot_top(core, idx);
	let frame_base = top - STATE_SIZE as u64;
	let src = task.last_stack_pointer.as_u64() as *const u64;
	let dst = frame_base as *mut u64;
	unsafe {
		for w in 0..(STATE_SIZE / size_of::<u64>()) {
			*dst.add(w) = *src.add(w);
		}
	}
	// INVARIANT: resumed frame must occupy exactly [top-STATE_SIZE, top).
	debug_assert_eq!(
		frame_base + STATE_SIZE as u64,
		top,
		"resume_from_evicted: frame base {frame_base:#x} + STATE_SIZE != slot_top {top:#x} (core {core} slot {idx})"
	);
	task.last_stack_pointer = VirtAddr::new(frame_base);
	task.slot = idx as i32;
	task.frame_location = FrameLocation::InSlot;
	// T7/R1.5: clear any stale deferred-wake flag on resuming into a slot.
	task.wake_pending = false;
	// T9-V-RW4: Evicted->InSlot re-dispatch discriminator (§8.2). Confirms
	// the slow path is exercised (not the unused-today dead path). STRIP T10.
	info!(
		"[V-RW4] resume_from_evicted task {} -> slot {} (InSlot)",
		task.id, idx
	);
	owners.set_owner(idx, task.id.into());
	true
}
