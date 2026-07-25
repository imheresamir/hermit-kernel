use alloc::boxed::Box;
use core::cell::Cell;
use core::ptr;
use core::sync::atomic::Ordering;

use aarch64_cpu::registers::{Readable, TPIDR_EL1, Writeable};
use async_executor::StaticLocalExecutor;
#[cfg(feature = "smp")]
use hermit_sync::InterruptTicketMutex;
use hermit_sync::{RawRwSpinLock, RawSpinMutex};

use super::CPU_ONLINE;
use super::interrupts::{IRQ_COUNTERS, IrqStatistics};
use crate::arch::aarch64::mm::paging::{BasePageSize, PageSize};
use crate::config::{ARCH_STATE_SIZE, DEFAULT_STACK_SIZE};
use crate::mm::{kernel_end_address, kernel_start_address};
#[cfg(feature = "smp")]
use crate::scheduler::SchedulerInput;
use crate::scheduler::{CoreId, PerCoreScheduler};

// Invariant guard: start.s D4 tail hardcodes `ldr x21, [x21, #24]` to load
// CoreLocal.scratch_slot, and the switch vectors hardcode `str x1, [x2, #24]`.
// If the CoreLocal repr(C) layout ever drifts, SP_EL1 would be loaded from the
// wrong offset (silently corrupting the per-task exception stack). Pin it.
const _: () = assert!(
	core::mem::offset_of!(CoreLocal, scratch_slot) == 24,
	"CoreLocal.scratch_slot must be at offset 24 (start.s D4 tail / switch vectors hardcode #24)"
);
const _: () = assert!(
	core::mem::offset_of!(CoreLocal, exception_sp) == 0,
	"CoreLocal.exception_sp must be at offset 0 (D4 tail: ldr x21,[x21,#0])"
);
const _: () = assert!(
	core::mem::offset_of!(CoreLocal, kernel_sp) == 16,
	"CoreLocal.kernel_sp must be at offset 16 (D5: ldr x9,[x9,#16]; switch: str x1,[x2,#16])"
);
const _: () = assert!(
	core::mem::offset_of!(CoreLocal, this) == 8,
	"CoreLocal.this must be at offset 8 (between exception_sp@0 and kernel_sp@16)"
);

// Linker-defined per-core exception-stack base (one slot per core, each
// `DEFAULT_STACK_SIZE +0x1000` (guard) wide; the top of slot N is E(N)).
// Same symbol §2.2 (start.rs boot block) and mod.rs::protect_stack_guards use.
unsafe extern "C" {
	static __start_exception_stacks: u8;
}

// Compile-time guard for D1's hardcoded `#288` immediate in start.s.
// `kernel_sp` is the SP_EL1 value after trap_exit pops the 18 stp pairs made by
// trap_entry. The creation-time stack marker is outside State and must not be
// added here; doing so advances kernel_sp by 16 bytes on every context switch.
// Cross-checks both the asm arithmetic (18 pairs × 16 B) and ARCH_STATE_SIZE.
const _: () = assert!(18 * 16 == ARCH_STATE_SIZE);

#[repr(C)]
pub(crate) struct CoreLocal {
	/// Per-core exception-stack top E (recomputed once install(), NOT read
	/// from live SP_EL1 — which has already descended by install time, §2.3.1 / §8.9).
	/// The §2.3.1 `trap_exit` tail reloads SP_EL1 =E from this field via
	/// `ldr x21,[x21,#0]` (offset 0 — first field, no repr-dependent offset).
	pub exception_sp: u64,
	/// Self-pointer (set in install()). Kept as 2nd field so `exception_sp`
	/// remains at offset 0 for the §2.3.1 trap_exit tail (`ldr x21,[x21,#0]`).
	this: *const Self,
	/// Current task's **kernel-stack top** (the task body runs EL1t on SP_EL0;
	/// the kernel stack is the EL1h switch frame + deep-kernel scratch area).
	/// Under Option D (INV-D) SP_EL1 is the per-core exception stack E, NOT the
	/// task kernel stack, so `call_with_kernel_stack` (and the D3 frame-copy
	/// dst computation) must reach the kernel stack explicitly via this field.
	/// Offset 16 (= 2×8): u64, 8-byte aligned. Updated on EVERY switch (D1/D3)
	/// before `trap_exit` runs, because a freshly-switched task body may call a
	/// `kernel_function` on its first instruction.
	pub(crate) kernel_sp: u64,
	/// Current task's **scratch-slot TOP** (per-task exception slot design).
	/// This is the top of the task's private exception scratch slot; the D4
	/// tail (start.s `trap_exit`) loads it into SP_EL1 on every EL1t return so
	/// the next EL1t exception builds its trap frame on the task's OWN slot
	/// instead of the shared per-core E. Offset 24 (= 3×8): u64. NOT to be
	/// conflated with `kernel_sp` (@16) — that stays the 128 KiB kernel-stack
	/// top for `call_with_kernel_stack`; overloading it with the ~5 KiB slot
	/// would overflow the handler stack (design D-1). Initialized at
	/// `install()` to this core's slot 0 top (idle/boot task's slot).
	scratch_slot: u64,
	/// ID of the current Core.
	core_id: CoreId,
	/// Scheduler of the current Core.
	scheduler: Cell<*mut PerCoreScheduler>,
	/// Interface to the interrupt counters
	irq_statistics: &'static IrqStatistics,
	/// The core-local async executor.
	ex: StaticLocalExecutor<RawSpinMutex, RawRwSpinLock>,
	/// Queues to handle incoming requests from the other cores
	#[cfg(feature = "smp")]
	pub scheduler_input: InterruptTicketMutex<SchedulerInput>,
}

impl CoreLocal {
	pub fn install() {
		let core_id = CPU_ONLINE.0.load(Ordering::Relaxed);

		// Recompute E (per-core exception-stack top), §2.3.1 — same formula as
		// §2.2 boot (start.rs) and mod.rs::protect_stack_guards.2a1t6
		// E(core) = &__start_exception_stacks +core*(DEFAULT_STACK_SIZE+0x1000)+DEFAULT_STACK_SIZE
		// (0x1000 =BasePageSize::SIZE guard per slot). Exception stack is DEFAULT_STACK_SIZE
		// (128KiB) -- a SCRATCH stack for trap_entry + dispatch (per design §1.1); the deep
		// handler work runs on the task's own kernel stack, not here. Sizing is via
		// link.x + start.rs + core_local.rs + protect_stack_guards (all 128KiB), kept
		// consistent; do NOT widen one without the other (see OPEN-2 re: measuring depth).
		// NOTE: the `.exception_stacks` linker section (crates/rs6/link.x) is
		// sized for ONE core by default; the supported core count is DERIVED
		// at runtime from the section size (mod.rs `linker_supported_cores`)
		// and clamped against the device tree via `max_bootable_cores()` =
		// min(dt_cpus, linker_supported_cores). There is NO MAX_CORES
		// constant anywhere — if the build grows the per-core sections, the
		// kernel observes the larger size and supports more cores. The boot
		// path must not depend on LIEF. Cores beyond the supported count are
		// never PSCI-woken, so their out-of-bounds SP_EL1 is never
		// dereferenced.
		let e_top = {
			let base = &raw const __start_exception_stacks as usize;
			let stride = DEFAULT_STACK_SIZE + BasePageSize::SIZE as usize;
			(base + (core_id as usize) * stride + DEFAULT_STACK_SIZE) as u64
		};
		// Belt-and-suspenders against the janky-loader rebase bug (see §8.13):
		// `exception_sp` must lie inside the loaded kernel image
		// (kernel_start_address()..kernel_end_address()). A zero or link-only
		// (un-rebased) value would be unmapped and fault on the first IRQ taken
		// while a task ran -- exactly the 2a.2 boot hang. This assert fires at
		// install() instead of failing opaquely later.
		debug_assert!(
			e_top > kernel_start_address().as_u64(),
			"exception_sp below image base -- loader did not rebase __start_exception_stacks (add kernel_start_address() bias)"
		);
		debug_assert!(
			e_top < kernel_end_address().as_u64(),
			"exception_sp above image end -- __start_exception_stacks stride/offset wrong"
		);

		let irq_statistics = if core_id == 0 {
			static FIRST_IRQ_STATISTICS: IrqStatistics = IrqStatistics::new();
			&FIRST_IRQ_STATISTICS
		} else {
			&*Box::leak(Box::new(IrqStatistics::new()))
		};

		let this = Self {
			exception_sp: e_top,
			this: ptr::null_mut(),
			core_id,
			kernel_sp: e_top, // D1/D6: boot/idle default = E (per-core exception stack).
			// Updated to the incoming task's kernel-stack top on every
			// switch (start.s switch path).
			// Per-task exception slot design: `scratch_slot` holds the TOP of
			// this core's current task's scratch slot; the D4 tail loads it
			// into SP_EL1 on EL1t returns. At boot the current "task" is the
			// idle/boot task, whose effective slot is E (the per-core
			// exception stack) until a real slot pool is allocated
			// (per-task-exception-slot-design.md Step 1). Initializing to
			// e_top keeps boot correct; the switch path republishes the real
			// per-task slot top on the first dispatch.
			// TODO(slot-pool): once .exception_slots is allocated, set this
			// to `slot_pool_base(core_id) + SLOT_STRIDE * 0 + SLOT_SIZE`
			// (core 0's slot 0 top = idle task's slot) per doc V-D5.
			scratch_slot: e_top,
			scheduler: Cell::new(ptr::null_mut()),
			irq_statistics,
			ex: StaticLocalExecutor::new(),
			#[cfg(feature = "smp")]
			scheduler_input: InterruptTicketMutex::new(SchedulerInput::new()),
		};
		let this = if core_id == 0 {
			take_static::take_static! {
				static FIRST_CORE_LOCAL: Option<CoreLocal> =None;
			}
			FIRST_CORE_LOCAL.take().unwrap().insert(this)
		} else {
			this.add_irq_counter();
			Box::leak(Box::new(this))
		};
		this.this = ptr::from_ref(this);

		let addr = (&raw mut *this).expose_provenance();
		TPIDR_EL1.set(addr.try_into().unwrap());
	}

	#[inline]
	pub fn get() -> &'static Self {
		let addr = TPIDR_EL1.get().try_into().unwrap();
		let ptr = ptr::with_exposed_provenance(addr);
		unsafe { &*ptr }
	}

	/// NEW-1 (option-d-per-task-slot-rebased.md §10): the current task's scratch
	/// slot TOP (SP_EL1 the D4 tail stages on EL1t returns). Read-only accessor —
	/// `scratch_slot` stays private so its pinned @24 offset (start.s D4 tail /
	/// switch vectors) can't be widened accidentally.
	#[inline]
	pub fn scratch_slot(&self) -> u64 {
		self.scratch_slot
	}

	/// §4D: Update kernel_sp for the incoming task. Called from the scheduler
	/// BEFORE the asm switch path (start.s). The asm publishes scratch_slot
	/// (@24) but NOT kernel_sp (@16); call_with_kernel_stack reads kernel_sp
	/// to set SP_EL1 for deep handler work. This must be set per-task-switch.
	///
	/// SAFETY: the caller must ensure `sp` is a valid kernel-stack top for the
	/// task about to be dispatched. A stale value causes call_with_kernel_stack
	/// to run deep work on the wrong stack → overflow → silent corruption.
	pub fn set_kernel_sp(&self, sp: u64) {
		// kernel_sp is not behind a Cell (unlike scheduler/scheduler_input),
		// so we write through a volatile raw pointer. This is safe because:
		//  (a) only the scheduler on this core calls this, between ensure_slot
		//      and the asm switch — no concurrent writer,
		//  (b) the asm reads this AFTER the `bl get_last_stack_pointer` return,
		//      which is a compiler barrier (bl implies memory clobber).
		unsafe {
			let ptr = (self as *const Self as *mut Self)
				.cast::<u8>()
				.add(core::mem::offset_of!(Self, kernel_sp))
				.cast::<u64>();
			ptr.write_volatile(sp);
		}
	}

	pub fn add_irq_counter(&self) {
		IRQ_COUNTERS
			.lock()
			.insert(self.core_id, self.irq_statistics);
	}
}

#[inline]
pub(crate) fn core_id() -> CoreId {
	if cfg!(target_os = "none") {
		CoreLocal::get().core_id
	} else {
		0
	}
}

#[inline]
pub(crate) fn core_scheduler() -> &'static mut PerCoreScheduler {
	unsafe { CoreLocal::get().scheduler.get().as_mut().unwrap() }
}

/// Non-panicking variant of [`core_scheduler`] for exception / diagnostic
/// paths that can legitimately run BEFORE the per-core scheduler is installed
/// (early boot). `core_scheduler()`'s unwrap turns any pre-scheduler fault
/// into a recursive panic loop (panic -> re-fault -> panic) that destroys the
/// original diagnostics; fault handlers must use this and degrade gracefully.
#[inline]
pub(crate) fn try_core_scheduler() -> Option<&'static mut PerCoreScheduler> {
	unsafe { CoreLocal::get().scheduler.get().as_mut() }
}

pub(crate) fn ex() -> &'static StaticLocalExecutor<RawSpinMutex, RawRwSpinLock> {
	&CoreLocal::get().ex
}

pub(crate) fn set_core_scheduler(scheduler: *mut PerCoreScheduler) {
	CoreLocal::get().scheduler.set(scheduler);
}

pub(crate) fn increment_irq_counter(irq_no: u8) {
	CoreLocal::get().irq_statistics.inc(irq_no);
}
