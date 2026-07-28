use alloc::boxed::Box;
use core::arch::asm;
use core::cell::Cell;
use core::sync::atomic::Ordering;
use core::{mem, ptr};

use async_executor::StaticLocalExecutor;
#[cfg(feature = "smp")]
use hermit_sync::InterruptTicketMutex;
#[cfg(feature = "continuations")]
use hermit_sync::RawInterruptSpinMutex;
#[cfg(not(feature = "continuations"))]
use hermit_sync::RawSpinMutex;
use hermit_sync::RawRwSpinLock;

/// Executor lock types — see arch/aarch64/kernel/core_local.rs for the
/// rationale. Under `continuations` `ExSpin` becomes interrupt-masking
/// `RawInterruptSpinMutex` (docs/stackful-continuations.md §9 O1); `ExRw`
/// stays `RawRwSpinLock` (no interrupt-masking RW primitive exists in
/// hermit-sync). Default keeps `RawSpinMutex`/`RawRwSpinLock` (byte-identical
/// / inert).
#[cfg(feature = "continuations")]
type ExSpin = RawInterruptSpinMutex;
#[cfg(feature = "continuations")]
type ExRw = RawRwSpinLock;
#[cfg(not(feature = "continuations"))]
type ExSpin = RawSpinMutex;
#[cfg(not(feature = "continuations"))]
type ExRw = RawRwSpinLock;
use x86_64::VirtAddr;
use x86_64::registers::model_specific::GsBase;
use x86_64::structures::tss::TaskStateSegment;

use super::CPU_ONLINE;
use super::interrupts::{IRQ_COUNTERS, IrqStatistics};
#[cfg(feature = "smp")]
use crate::scheduler::SchedulerInput;
use crate::scheduler::{CoreId, PerCoreScheduler};

pub(crate) struct CoreLocal {
	this: *const Self,
	/// Sequential ID of this CPU Core.
	core_id: CoreId,
	/// Scheduler for this CPU Core.
	scheduler: Cell<*mut PerCoreScheduler>,
	/// Task State Segment (TSS) allocated for this CPU Core.
	pub tss: Cell<*mut TaskStateSegment>,
	/// Start address of the kernel stack
	pub kernel_stack: Cell<*mut u8>,
	/// Interface to the interrupt counters
	irq_statistics: &'static IrqStatistics,
	/// The core-local async executor.
	ex: StaticLocalExecutor<ExSpin, ExRw>,
	/// Queues to handle incoming requests from the other cores
	#[cfg(feature = "smp")]
	pub scheduler_input: InterruptTicketMutex<SchedulerInput>,
}

impl CoreLocal {
	pub fn install() {
		assert_eq!(VirtAddr::zero(), GsBase::read());

		let core_id = CPU_ONLINE.load(Ordering::Relaxed);

		let irq_statistics = if core_id == 0 {
			static FIRST_IRQ_STATISTICS: IrqStatistics = IrqStatistics::new();
			&FIRST_IRQ_STATISTICS
		} else {
			&*Box::leak(Box::new(IrqStatistics::new()))
		};

		let this = Self {
			this: ptr::null_mut(),
			core_id,
			scheduler: Cell::new(ptr::null_mut()),
			tss: Cell::new(ptr::null_mut()),
			kernel_stack: Cell::new(ptr::null_mut()),
			irq_statistics,
			ex: StaticLocalExecutor::new(),
			#[cfg(feature = "smp")]
			scheduler_input: InterruptTicketMutex::new(SchedulerInput::new()),
		};
		let this = if core_id == 0 {
			take_static::take_static! {
				static FIRST_CORE_LOCAL: Option<CoreLocal> = None;
			}
			FIRST_CORE_LOCAL.take().unwrap().insert(this)
		} else {
			this.add_irq_counter();
			Box::leak(Box::new(this))
		};
		this.this = ptr::from_ref(this);

		GsBase::write(VirtAddr::from_ptr(this));
	}

	#[inline]
	pub fn get() -> &'static Self {
		debug_assert_ne!(VirtAddr::zero(), GsBase::read());
		unsafe {
			let raw: *const Self;
			asm!("mov {}, gs:{}", out(reg) raw, const mem::offset_of!(Self, this), options(nomem, nostack, preserves_flags));
			&*raw
		}
	}

	pub fn add_irq_counter(&self) {
		IRQ_COUNTERS
			.lock()
			.insert(self.core_id, self.irq_statistics);
	}
}

pub(crate) fn core_id() -> CoreId {
	if cfg!(target_os = "none") {
		CoreLocal::get().core_id
	} else {
		0
	}
}

pub(crate) fn core_scheduler() -> &'static mut PerCoreScheduler {
	unsafe { CoreLocal::get().scheduler.get().as_mut().unwrap() }
}

pub(crate) fn ex() -> &'static StaticLocalExecutor<ExSpin, ExRw> {
	&CoreLocal::get().ex
}

pub(crate) fn set_core_scheduler(scheduler: *mut PerCoreScheduler) {
	CoreLocal::get().scheduler.set(scheduler);
}

pub(crate) fn increment_irq_counter(irq_no: u8) {
	CoreLocal::get().irq_statistics.inc(irq_no);
}
