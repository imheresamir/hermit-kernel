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
#[cfg(feature = "smp")]
use crate::scheduler::SchedulerInput;
use crate::scheduler::{CoreId, PerCoreScheduler};
use crate::arch::aarch64::mm::paging::{BasePageSize, PageSize};
use crate::config::DEFAULT_STACK_SIZE;
use crate::mm::{kernel_end_address, kernel_start_address};

// Linker-defined per-core exception-stack base (one slot per core, each
// `DEFAULT_STACK_SIZE +0x1000` (guard) wide; the top of slot N is E(N)).
// Same symbol §2.2 (start.rs boot block) and mod.rs::protect_stack_guards use.
unsafe extern "C" {
	static __start_exception_stacks: u8;
}

pub(crate) struct CoreLocal {
	/// Per-core exception-stack top E (recomputed once install(), NOT read
	/// from live SP_EL1 — which has already descended by install time, §2.3.1 / §8.9).
	/// The §2.3.1 `trap_exit` tail reloads SP_EL1 =E from this field via
	/// `ldr x21,[x21,#0]` (offset 0 — first field, no repr-dependent offset).
	pub exception_sp: u64,
	/// Self-pointer (set in install()). Kept as 2nd field so `exception_sp`
	/// remains at offset 0 for the §2.3.1 trap_exit tail (`ldr x21,[x21,#0]`).
	this: *const Self,
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
		let core_id =CPU_ONLINE.0.load(Ordering::Relaxed);

		// Recompute E (per-core exception-stack top), §2.3.1 — same formula as
		// §2.2 boot (start.rs) and mod.rs::protect_stack_guards.2a1t6
		// E(core) = &__start_exception_stacks +core*(DEFAULT_STACK_SIZE+0x1000)+DEFAULT_STACK_SIZE
		// (0x1000 =BasePageSize::SIZE guard per slot). NOT read from live SP_EL1
		// (descended by install time).
		let e_top = {
			// &__start_exception_stacks: for a REAL (non-INSERT) section the
			// loader rebases this symbol to the runtime PIE base, so it already
			// yields the runtime address (same as the boot block's `sym` form).
			// The debug_assert below verifies it lands inside the image; if a
			// future loader/section change stops rebasing it, the lower-bound
			// assert fires instead of silently faulting on the first IRQ.
			let base = &raw const __start_exception_stacks as usize;
			let stride =DEFAULT_STACK_SIZE +BasePageSize::SIZE as usize;
			(base + (core_id as usize) *stride +DEFAULT_STACK_SIZE) as u64
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

		let irq_statistics = if core_id ==0 {
			static FIRST_IRQ_STATISTICS: IrqStatistics =IrqStatistics::new();
			&FIRST_IRQ_STATISTICS
		} else {
			&*Box::leak(Box::new(IrqStatistics::new()))
		};

		let this =Self {
			exception_sp: e_top,
			this: ptr::null_mut(),
			core_id,
			scheduler: Cell::new(ptr::null_mut()),
			irq_statistics,
			ex: StaticLocalExecutor::new(),
			#[cfg(feature = "smp")]
			scheduler_input: InterruptTicketMutex::new(SchedulerInput::new()),
		};
		let this = if core_id ==0 {
			take_static::take_static! {
				static FIRST_CORE_LOCAL: Option<CoreLocal> =None;
			}
			FIRST_CORE_LOCAL.take().unwrap().insert(this)
		} else {
			this.add_irq_counter();
			Box::leak(Box::new(this))
		};
		this.this =ptr::from_ref(this);

		let addr =(&raw mut *this).expose_provenance();
		TPIDR_EL1.set(addr.try_into().unwrap());
	}

	#[inline]
	pub fn get() -> &'static Self {
		let addr = TPIDR_EL1.get().try_into().unwrap();
		let ptr = ptr::with_exposed_provenance(addr);
		unsafe { &*ptr }
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

pub(crate) fn ex() -> &'static StaticLocalExecutor<RawSpinMutex, RawRwSpinLock> {
	&CoreLocal::get().ex
}

pub(crate) fn set_core_scheduler(scheduler: *mut PerCoreScheduler) {
	CoreLocal::get().scheduler.set(scheduler);
}

pub(crate) fn increment_irq_counter(irq_no: u8) {
	CoreLocal::get().irq_statistics.inc(irq_no);
}
