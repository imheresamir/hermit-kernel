use alloc::collections::{BTreeMap, VecDeque};
use core::arch::asm;
use core::ptr::{self, NonNull};
use core::sync::atomic::{AtomicU64, Ordering};

use aarch64_cpu::asm::barrier::{ISH, SY, dmb, isb};
use aarch64_cpu::registers::*;
use ahash::RandomState;
use arm_gic::gicv3::{GicCpuInterface, GicV3};
use arm_gic::{IntId, InterruptGroup, Trigger, UniqueMmioPointer};
use fdt::standard_nodes::Compatible;
use free_list::PageLayout;
use hashbrown::HashMap;
use hermit_sync::{InterruptSpinMutex, InterruptTicketMutex, OnceCell, SpinMutex};
use memory_addresses::{PhysAddr, VirtAddr};

use crate::arch::aarch64::kernel::core_local::{core_id, core_scheduler, increment_irq_counter};
use crate::arch::aarch64::kernel::scheduler::State;
use crate::arch::aarch64::kernel::serial::handle_uart_interrupt;
use crate::arch::aarch64::mm::paging::{self, BasePageSize, PageSize, PageTableEntryFlags};
use crate::drivers::InterruptHandlerMap;
use crate::env;
use crate::mm::{PageAlloc, PageRangeAllocator};
use crate::scheduler::{self, CoreId, timer_interrupts};

/// The ID of the first Private Peripheral Interrupt.
const PPI_START: u8 = 16;
/// The ID of the first Shared Peripheral Interrupt.
#[allow(dead_code)]
const SPI_START: u8 = 32;
/// Software-generated interrupt for rescheduling
pub(crate) const SGI_RESCHED: u8 = 1;

/// Number of the timer interrupt
static mut TIMER_INTERRUPT: u32 = 0;
/// Number of the UART interrupt
static mut UART_INTERRUPT: u32 = 0;
/// Possible interrupt handlers
static INTERRUPT_HANDLERS: OnceCell<InterruptHandlerMap> = OnceCell::new();
/// Driver for the Arm Generic Interrupt Controller version 3 (or 4).
pub(crate) static GIC: SpinMutex<Option<GicV3<'_>>> = SpinMutex::new(None);

/// Enable all interrupts
#[inline]
pub fn enable() {
	dmb(ISH);
	unsafe {
		asm!(
			"msr daifclr, {mask}",
			mask = const 0b111,
			options(nostack),
		);
	}
	dmb(ISH);
}

/// Enable all interrupts and wait for the next interrupt (wfi instruction)
#[inline]
pub fn enable_and_wait() {
	dmb(ISH);
	unsafe {
		asm!(
			"msr daifclr, {mask}; wfi",
			mask = const 0b111,
			options(nostack),
		);
	}
	dmb(ISH);
}

/// Disable all interrupts
#[inline]
pub fn disable() {
	dmb(ISH);
	unsafe {
		asm!(
			"msr daifset, {mask}",
			mask = const 0b111,
			options(nostack),
		);
	}
	dmb(ISH);
}

pub(crate) fn install_handlers(old_handlers: InterruptHandlerMap) {
	let mut handlers: InterruptHandlerMap =
		HashMap::with_hasher(RandomState::with_seeds(0, 0, 0, 0));
	fn timer_handler() {
		debug!("Handle timer interrupt");
		timer_interrupts::clear_active_and_set_next();
	}

	for (key, value) in old_handlers.into_iter() {
		handlers.insert(key + SPI_START, value);
	}

	unsafe {
		if let Some(queue) = handlers.get_mut(&(u8::try_from(TIMER_INTERRUPT).unwrap() + PPI_START))
		{
			queue.push_back(timer_handler);
		} else {
			let mut queue = VecDeque::<fn()>::new();
			queue.push_back(timer_handler);
			handlers.insert(u8::try_from(TIMER_INTERRUPT).unwrap() + PPI_START, queue);
		}

		if let Some(queue) = handlers.get_mut(&(u8::try_from(UART_INTERRUPT).unwrap() + SPI_START))
		{
			queue.push_back(handle_uart_interrupt);
		} else {
			let mut queue = VecDeque::<fn()>::new();
			queue.push_back(handle_uart_interrupt);
			handlers.insert(u8::try_from(UART_INTERRUPT).unwrap() + SPI_START, queue);
		}
	}

	INTERRUPT_HANDLERS.set(handlers).unwrap();
}

#[unsafe(no_mangle)]
pub(crate) extern "C" fn do_fiq(_state: &State) -> *mut usize {
	let Some(irqid) = GicCpuInterface::get_and_acknowledge_interrupt(InterruptGroup::Group1) else {
		return ptr::null_mut();
	};

	let vector: u8 = u32::from(irqid).try_into().unwrap();

	debug!("Receive fiq {vector}");
	increment_irq_counter(vector);

	if let Some(handlers) = INTERRUPT_HANDLERS.get()
		&& let Some(queue) = handlers.get(&vector)
	{
		for handler in queue.iter() {
			handler();
		}
	}
	// Part B: on the exception stack E, only do the bounded wake-move
	// (ready_queue update); the async executor is drained off E by the reactor
	// idle loop (PerCoreScheduler::run). Do NOT call handle_waiting_tasks()
	// here -- it runs executor::run() on the current stack (= E post-flip).
	core_scheduler().wake_pending_tasks();

	GicCpuInterface::end_interrupt(irqid, InterruptGroup::Group1);

	core_scheduler().scheduler(false).unwrap_or_default()
}

#[unsafe(no_mangle)]
pub(crate) extern "C" fn do_irq(_state: &State) -> *mut usize {
	// [MEASURE] peak E usage: read SP_EL1 (current depth on E) and compare to
	// exception_sp (E top). delta = bytes consumed by this point in the handler.
	// NOTE: `mrs {}, sp_el1` is UNDEFINED at EL1 (SP_EL1 is EL2/EL3-readable only)
	// and traps as EC=0x0 (Unknown) -- do_irq runs EL1h (SPSEL=1) so `sp` aliases
	// SP_EL1; read it with `mov` (same workaround as do_sync's error handler).
	let sp_el1: u64;
	unsafe { core::arch::asm!("mov {val}, sp", val = out(reg) sp_el1) };
	let e_top = crate::arch::aarch64::kernel::core_local::CoreLocal::get().exception_sp as u64;
	// Only meaningful when this IRQ was taken from the exception stack E
	// (e_top is the top of E; sp_el1 on E is < e_top). In the current EL1h
	// model, task IRQs instead land on the task's kernel stack (sp_el1 > e_top),
	// so skip those to avoid underflow and to isolate the peak E depth.
	if sp_el1 < e_top {
		let used = e_top - sp_el1;
		debug!("[E-DEPTH] do_irq on E: sp_el1={:#x} exception_sp={:#x} used={:#x} ({} bytes)", sp_el1, e_top, used, used);
	}
	let Some(irqid) = GicCpuInterface::get_and_acknowledge_interrupt(InterruptGroup::Group1) else {
		return ptr::null_mut();
	};

	let vector: u8 = u32::from(irqid).try_into().unwrap();

	debug!("Receive interrupt {vector}");
	increment_irq_counter(vector);

	if let Some(handlers) = INTERRUPT_HANDLERS.get()
		&& let Some(queue) = handlers.get(&vector)
	{
		for handler in queue.iter() {
			handler();
		}
	}
	// Part B: on the exception stack E, only do the bounded wake-move
	// (ready_queue update); the async executor is drained off E by the reactor
	// idle loop (PerCoreScheduler::run). Do NOT call handle_waiting_tasks()
	// here -- it runs executor::run() on the current stack (= E post-flip).
	core_scheduler().wake_pending_tasks();

	GicCpuInterface::end_interrupt(irqid, InterruptGroup::Group1);

	let result = core_scheduler().scheduler(false).unwrap_or_default();

	// === INSTRUMENTATION: log context switch result ===
	if !result.is_null() {
		// result = ptr to old task's last_stack_pointer field
		// *result = old task's SP_EL1 (the trap_entry frame on E)
		let old_sp = unsafe { *result };
		let new_task_id = core_scheduler().get_current_task_id();
		let new_lsp = core_scheduler().get_last_stack_pointer();
		warn!("[TRACE-IRQ] ctx switch: old_sp={old_sp:#x} new_task={new_task_id:?} new_lsp={:#x}", new_lsp.as_u64());
		// Dump the new task's State at new_lsp
		let state_ptr = new_lsp.as_usize() as *const u64;
		let raw = unsafe { core::slice::from_raw_parts(state_ptr, 36) };
		warn!("[TRACE-IRQ] new State[0]={:#x} [1]={:#x} [8]={:#x} [35]={:#x}",
			raw[0], raw[1], raw[8], raw[35]);
	}

	result
}

#[unsafe(no_mangle)]
pub(crate) extern "C" fn do_sync(state: &State) {
	let esr = ESR_EL1.get();
	let ec_raw = ESR_EL1.read(ESR_EL1::EC);
	let ec: ESR_EL1::EC::Value = ESR_EL1.read_as_enum(ESR_EL1::EC).unwrap();
	let iss = ESR_EL1.read(ESR_EL1::ISS);
	let pc = ELR_EL1.get();

	/* data abort from lower or current level */
	if (ec == ESR_EL1::EC::Value::DataAbortCurrentEL)
		|| (ec == ESR_EL1::EC::Value::DataAbortLowerEL)
	{
		/* check if value in far_el1 is valid */
		if (iss & (1 << 10)) == 0 {
			/* read far_el1 register, which holds the faulting virtual address */
			let far = FAR_EL1.get();

			// add page fault handler

			error!("Current stack pointer {state:p}");
			error!("Unable to handle page fault at {far:#x}");
			error!("Exception return address {:#x}", ELR_EL1.get());
			error!("Thread ID register {:#x}", TPIDR_EL0.get());
			error!("Table Base Register {:#x}", TTBR0_EL1.get());
			error!("Exception Syndrome Register {esr:#x}");

			// === EL1t diagnostic: what writes to kernel_stack_top? ===
			// state.sp_el0 is SP_EL0 captured at fault time (frame slot @24).
			// far is the faulting VA. Compare both against the current
			// task's computed kernel_stack_top = base + size.
			let sp_el0_fault = state.sp_el0;
			let cur_id = core_scheduler().get_current_task_id();
			let ktop = core_scheduler().get_current_task_kernel_stack_top();
			error!(
				"DIAG sp_el0_fault={sp_el0_fault:#x} far={far:#x} cur_task={cur_id:?}"
			);
			error!(
				"DIAG kernel_stack_top={ktop:#x} (base+KERNEL_STACK_SIZE)"
			);
			// Runtime image base (rebased). link_pc = fault_pc - base lets us
			// map the fault to a function in the (link-addressed) binary via
			// host objdump. kernel_start_address() == executable_start() rebased.
			let base = crate::mm::kernel_start_address().as_u64();
			let link_pc = ELR_EL1.get().wrapping_sub(base);
			error!(
				"DIAG kernel_start_address={base:#x} fault_pc(runtime)={:#x} link_pc={link_pc:#x}",
				ELR_EL1.get()
			);
			error!(
				"DIAG sp_el0==ktop? {} far==ktop? {} sp_el0==far? {}",
				sp_el0_fault == ktop,
				far == ktop,
				sp_el0_fault == far
			);

			if let Some(irqid) =
				GicCpuInterface::get_and_acknowledge_interrupt(InterruptGroup::Group1)
			{
				GicCpuInterface::end_interrupt(irqid, InterruptGroup::Group1);
			} else {
				error!("Unable to acknowledge interrupt!");
			}

			error!("Fatal: halting in data-abort handler to surface the real fault.");
			// DEBUG SURFACE: previously this called `scheduler::abort()`
			// (`core_scheduler().exit(-1)`), whose panic/shutdown path walks the
			// (very deep) application call stack and overflows the exception
			// stack, triple-faulting and masking the original fault. Spin
			// instead so the FAR/ELR/ESR printed above survive. The real bug
			// is the application fault (observed PC = `sparse_chunk::insert`
			// in Gleam/HAMT code at client connect) - fix that, then restore
			// `scheduler::abort()`.
			loop {
				core::hint::spin_loop();
			}
		} else {
			error!("Unknown exception");
		}
	} else if ec == ESR_EL1::EC::Value::Brk64 {
		error!("Trap to debugger, PC={pc:#x}");
		loop {
			core::hint::spin_loop();
		}
	} else if ec_raw == 0x20 || ec_raw == 0x21 {
		// Instruction Abort from lower (0x20) or current (0x21) EL
		let far = FAR_EL1.get();
		let sp_val: u64;
		// At EL1h the running stack pointer IS `sp` (SP_EL1). Reading the
		// `SP_EL1` *system register* via `mrs` is UNDEFINED at EL1 (only
		// accessible from EL2/EL3) and traps as an Undefined Instruction —
		// which re-enters el1_sync and causes an infinite exception storm.
		unsafe { core::arch::asm!("mov {val}, sp", val = out(reg) sp_val) };
		error!("Instruction abort at {far:#x}, PC={pc:#x}, EC={ec_raw:#x}");
		error!("Current stack pointer {state:p}, SP_EL1={sp_val:#x}");
		error!("Exception Syndrome Register {esr:#x}");
		error!("Thread ID register {:#x}", TPIDR_EL0.get());
		let (sx29, sx30) = (state.x29, state.x30);
		error!("State x29(fp)={sx29:#x} x30(lr)={sx30:#x}");

		// === INSTRUMENTATION: dump full trap_entry State from emergency stack ===
		// state is the #[repr(C, packed)] trap_entry frame. Copy fields to
		// locals to avoid unaligned-reference errors.
		let state_addr = state as *const State as u64;
		let (sx0, sx1, sx25, sx30_val, sspsel, sspsr, stpidr, ssp_el0) = (
			state.x0, state.x1, state.x25, state.x30, state.spsel, state.spsr_el1, state.tpidr_el0, state.sp_el0
		);
		let s_elr = state.elr_el1 as *const () as u64;
		error!("[TRACE-SYNC] State @ {state_addr:#x}: elr_el1={s_elr:#x} x0={sx0:#x} x1={sx1:#x} x25={sx25:#x} x30={sx30_val:#x} spsel={sspsel:#x} spsr={sspsr:#x} tpidr={stpidr:#x} sp_el0={ssp_el0:#x}");

		let task_id = core_scheduler().get_current_task_id();
		error!("Crashed in task {task_id:?}");

		// === INSTRUMENTATION: read raw State words for cross-check ===
		let raw = unsafe { core::slice::from_raw_parts(state as *const State as *const u64, 36) };
		error!("[TRACE-SYNC] raw[0]={:#x} raw[1]={:#x} raw[8]={:#x} raw[35]={:#x}",
			raw[0], raw[1], raw[8], raw[35]);

		// === INSTRUMENTATION: check task_start trace buffer ===
		unsafe {
			let trace = crate::arch::aarch64::kernel::scheduler::TASK_START_TRACE;
			error!("[TRACE-SYNC] TASK_START_TRACE[8]={:#x} (0x42=reached) [0]={:#x} [1]={:#x} [2]={:#x} [3]={:#x}",
				trace[8], trace[0], trace[1], trace[2], trace[3]);
			error!("[TRACE-SYNC] TASK_START_TRACE SP_after_spsel={:#x} SP_after_func={:#x}",
				trace[4], trace[5]);
		}

		// === INSTRUMENTATION: dump stack memory around SP_EL0 and FP ===
		// SP_EL0 at fault time tells us where the function's stack was.
		// Dump from (SP_EL0 - 16) through (initial_SP + 16) to see the full frame.
		let sp_el0_fault = state.sp_el0;
		let fp_fault = sx29;
		if sp_el0_fault != 0 {
			// Use TASK_START_TRACE[4] as the initial SP_EL0 (set in task_start after spsel)
			let initial_sp_el0: u64 = unsafe { crate::arch::aarch64::kernel::scheduler::TASK_START_TRACE[4] };
			// Sentinel-encoding helper (mirror of create_stack_frame):
			// sentinel(w) = 0x5EED_0000_0000_0000 | (w & 0x0000_FFFF_FFFF_FFF8).
			let is_sentinel = |w: u64, addr: u64| -> bool {
				(w & 0xFFFF_0000_0000_0000) == 0x5EED_0000_0000_0000
					&& (w & 0x0000_FFFF_FFFF_FFFF) == (addr & 0x0000_FFFF_FFFF_FFF8)
			};
			// Classify: a window around SP_EL0. Report *how many* slots
			// broke and list the first/last broken VA so the footprint
			// and clobbered-slot is derivable from one line.
			let guard_page = if initial_sp_el0 > 0 { initial_sp_el0 + 16 } else { sp_el0_fault + 0x200 };
			let lo = sp_el0_fault.saturating_sub(16);
			let hi = guard_page.max(sp_el0_fault + 8);
			let nwords = ((hi - lo) / 8) as usize;
			if nwords > 0 && nwords <= 128 {
				let words = unsafe { core::slice::from_raw_parts(lo as *const u64, nwords) };
				error!("[TRACE-SYNC] STACK DUMP lo={lo:#x} hi={hi:#x} ({} words):", nwords);
				let mut i = 0;
				let mut first_broke: Option<u64> = None;
				let mut last_broke: Option<u64> = None;
				let mut broke_count = 0usize;
				while i < nwords {
					let addr = lo + (i as u64) * 8;
					let mut line = [0u64; 4];
					let count = nwords - i.min(nwords);
					let show = count.min(4);
					for j in 0..show {
						let v = words[i + j];
						let a = addr + (j as u64) * 8;
						line[j] = v;
						// A slot is "broken" if it was a sentinel and now
						// isn't. Value 0 == zeroed; text addr == displaced
						// real LR; else foreign.
						if is_sentinel(v, a) {
							// intact
						} else {
							broke_count += 1;
							if first_broke.is_none() { first_broke = Some(a); }
							last_broke = Some(a);
						}
					}
					error!("  {:#x}: {:#x} {:#x} {:#x} {:#x}",
						addr, line[0], line[1], line[2], line[3]);
					i += show;
				}
				if let (Some(f), Some(l)) = (first_broke, last_broke) {
					error!("[TRACE-SYNC] SENTINEL broken: count={} first_broke_va={:#x} last_broke_va={:#x} span={} bytes",
						broke_count, f, l, l - f + 8);
				} else {
					error!("[TRACE-SYNC] SENTINEL intact: no clobber in dumped window");
				}
			}
		}
		if fp_fault != 0 && fp_fault >= sp_el0_fault && fp_fault < 0x800015d92000 {
			let fp_words = unsafe { core::slice::from_raw_parts(fp_fault as *const u64, 4) };
			error!("[TRACE-SYNC] Frame @ FP={fp_fault:#x}: [FP_chain={:#x} LR={:#x} {:#x} {:#x}]",
				fp_words[0], fp_words[1], fp_words[2], fp_words[3]);
		}

		scheduler::abort()
	} else if ec == ESR_EL1::EC::Value::TrappedFP {
		trace!("Floating point trap");

		// We disabled FPU traps to lazily save the FPU state
		// This synchronous exception is triggered when floating point is used
		// So now save and restore the FPU state
		CPACR_EL1.modify(CPACR_EL1::FPEN::TrapNothing);
		isb(SY);

		// Let the scheduler set up the FPU for the current task
		core_scheduler().fpu_switch();
	} else {
		let far = FAR_EL1.get();
		let sp_val: u64;
		// See note above: `mrs ..., sp_el1` is UNDEFINED at EL1. Use `sp`.
		unsafe { core::arch::asm!("mov {val}, sp", val = out(reg) sp_val) };
		error!("Unsupported exception class: {ec_raw:#x}, PC={pc:#x}, FAR={far:#x}");
		error!("SP_EL1={sp_val:#x}");
		// State is #[repr(C, packed)] -- copy fields to locals to avoid misaligned references.
		let (sx0, sx1, sx2, sx29, sx30) = (state.x0, state.x1, state.x2, state.x29, state.x30);
		error!("State x0={sx0:#x} x1={sx1:#x} x2={sx2:#x} x30(lr)={sx30:#x} x29(fp)={sx29:#x}");
		let task_id = core_scheduler().get_current_task_id();
		error!("Crashed in task {task_id:?}");

		loop {
			core::hint::spin_loop();
		}
	}
}

#[unsafe(no_mangle)]
pub(crate) extern "C" fn do_bad_mode(_state: &State, reason: u32) -> ! {
	error!("Receive unhandled exception: {reason}");

	scheduler::abort()
}

#[unsafe(no_mangle)]
pub(crate) extern "C" fn do_error(_state: &State) -> ! {
	error!("Receive error interrupt");

	scheduler::abort()
}

/// Send a Software Generated Interrupt to a specific core.
///
/// Bypasses the arm-gic crate's `GicCpuInterface::send_sgi()` which has an ABI bug:
/// `Result<(), GicError>` is returned via a hidden pointer in x8, but the caller
/// passes NULL, so the callee always crashes writing `Ok(())` through null.
///
/// Instead, we build the ICC_SGI1R_EL1 register value manually and issue the MSR
/// directly. This is safe, efficient, and avoids the broken ABI entirely.
///
/// ICC_SGI1R_EL1 encoding:
///   bits [63:48] = Aff3
///   bits [43:40] = IRM (1 = exclude self)
///   bits [39:32] = Aff2
///   bits [31:24] = INTID
///   bits [23:16] = Aff1
///   bits [15:0]  = TargetList
pub fn wakeup_core(core_id: CoreId) {
	debug!("Wakeup core {core_id}");

	let intid: u64 = u64::from(SGI_RESCHED);
	let target_list: u64 = 1u64 << u64::from(core_id);
	let sgi_value: u64 = (intid << 24) | target_list;

	// SAFETY: ICC_SGI1R_EL1 is a system register that triggers an SGI to the
	// specified cores. Writing to it is safe as long as the GIC CPU interface is
	// initialized (which it is — we only get here after interrupts::init_cpu).
	unsafe {
		core::arch::asm!(
			"msr ICC_SGI1R_EL1, {value:x}",
			value = in(reg) sgi_value,
			options(nostack),
		);
	}
}

pub(crate) fn init() {
	info!("Initialize generic interrupt controller");

	let fdt = env::fdt().unwrap();

	let intc_node = fdt.find_node("/intc").unwrap();
	let mut reg_iter = intc_node.reg().unwrap();
	let gicd_reg = reg_iter.next().unwrap();
	let gicr_reg = reg_iter.next().unwrap();
	let gicd_start = PhysAddr::from(gicd_reg.starting_address.addr());
	let gicr_start = PhysAddr::from(gicr_reg.starting_address.addr());
	let gicd_size = u64::try_from(gicd_reg.size.unwrap()).unwrap();
	let gicr_size = u64::try_from(gicr_reg.size.unwrap()).unwrap();

	let num_cpus = fdt.cpus().count();

	let cpu_id: usize = core_id().try_into().unwrap();

	let compatible = intc_node
		.compatible()
		.map(Compatible::first)
		.unwrap_or("unknown");
	let is_gic_v4 = if compatible == "arm,gic-v4" {
		info!("Found GIC v4 with {num_cpus} cpus");
		true
	} else if compatible == "arm,gic-v3" {
		info!("Found GIC v3 with {num_cpus} cpus");
		false
	} else {
		panic!("{compatible} isn't supported")
	};

	info!("Found GIC Distributor interface at {gicd_start:p} (size {gicd_size:#X})");
	info!(
		"Found generic interrupt controller redistributor at {gicr_start:p} (size {gicr_size:#X})"
	);

	let layout = PageLayout::from_size_align(gicd_size.try_into().unwrap(), 0x10000).unwrap();
	let page_range = PageAlloc::allocate(layout).unwrap();
	let gicd_address = VirtAddr::from(page_range.start());
	debug!("Mapping GIC Distributor interface to virtual address {gicd_address:p}");

	let mut flags = PageTableEntryFlags::empty();
	flags.device().writable().execute_disable();
	paging::map::<BasePageSize>(
		gicd_address,
		gicd_start,
		(gicd_size / BasePageSize::SIZE).try_into().unwrap(),
		flags,
	);

	let layout = PageLayout::from_size_align(gicr_size.try_into().unwrap(), 0x10000).unwrap();
	let page_range = PageAlloc::allocate(layout).unwrap();
	let gicr_address = VirtAddr::from(page_range.start());
	debug!("Mapping generic interrupt controller to virtual address {gicr_address:p}");
	paging::map::<BasePageSize>(
		gicr_address,
		gicr_start,
		(gicr_size / BasePageSize::SIZE).try_into().unwrap(),
		flags,
	);

	let gicd = unsafe { UniqueMmioPointer::new(NonNull::new(gicd_address.as_mut_ptr()).unwrap()) };
	let gicr = NonNull::new(gicr_address.as_mut_ptr()).unwrap();

	let mut gic = unsafe { GicV3::new(gicd, gicr, num_cpus, is_gic_v4) };
	gic.setup(cpu_id);
	GicCpuInterface::set_priority_mask(0xff);

	if let Some(timer_node) = fdt.find_compatible(&["arm,armv8-timer", "arm,armv7-timer"]) {
		let irq_slice = timer_node.property("interrupts").unwrap().value;

		/* Secure Phys IRQ */
		let (_irqtype, irq_slice) = irq_slice.split_at(size_of::<u32>());
		let (_irq, irq_slice) = irq_slice.split_at(size_of::<u32>());
		let (_irqflags, irq_slice) = irq_slice.split_at(size_of::<u32>());
		/* Non-secure Phys IRQ */
		let (irqtype, irq_slice) = irq_slice.split_at(size_of::<u32>());
		let (irq, irq_slice) = irq_slice.split_at(size_of::<u32>());
		let (irqflags, _irq_slice) = irq_slice.split_at(size_of::<u32>());
		let irqtype = u32::from_be_bytes(irqtype.try_into().unwrap());
		let irq = u32::from_be_bytes(irq.try_into().unwrap());
		let irqflags = u32::from_be_bytes(irqflags.try_into().unwrap());
		unsafe {
			TIMER_INTERRUPT = irq;
		}

		debug!("Timer interrupt: {irq}, type {irqtype}, flags {irqflags}");

		IRQ_NAMES
			.lock()
			.insert(u8::try_from(irq).unwrap() + PPI_START, "Timer");

		// enable timer interrupt
		let timer_irqid = if irqtype == 1 {
			IntId::ppi(irq)
		} else if irqtype == 0 {
			IntId::spi(irq)
		} else {
			panic!("Invalid interrupt type");
		};
		gic.set_interrupt_priority(timer_irqid, Some(cpu_id), 0x00)
			.unwrap();
		if (irqflags & 0xf) == 4 || (irqflags & 0xf) == 8 {
			gic.set_trigger(timer_irqid, Some(cpu_id), Trigger::Level)
				.unwrap();
		} else if (irqflags & 0xf) == 2 || (irqflags & 0xf) == 1 {
			gic.set_trigger(timer_irqid, Some(cpu_id), Trigger::Edge)
				.unwrap();
		} else {
			panic!("Invalid interrupt level!");
		}
		gic.enable_interrupt(timer_irqid, Some(cpu_id), true)
			.unwrap();
	}

	if let Some(uart_node) = fdt.find_compatible(&["arm,pl011"]) {
		let irq_slice = uart_node.property("interrupts").unwrap().value;
		let (irqtype, irq_slice) = irq_slice.split_at(size_of::<u32>());
		let (irq, irq_slice) = irq_slice.split_at(size_of::<u32>());
		let (irqflags, _) = irq_slice.split_at(size_of::<u32>());
		let irqtype = u32::from_be_bytes(irqtype.try_into().unwrap());
		let irq = u32::from_be_bytes(irq.try_into().unwrap());
		let irqflags = u32::from_be_bytes(irqflags.try_into().unwrap());

		unsafe {
			UART_INTERRUPT = irq;
		}

		debug!("UART interrupt: {irq}, type {irqtype}, flags {irqflags}");

		IRQ_NAMES
			.lock()
			.insert(u8::try_from(irq).unwrap() + SPI_START, "UART");

		// enable uart interrupt
		let uart_irqid = if irqtype == 1 {
			IntId::ppi(irq)
		} else if irqtype == 0 {
			IntId::spi(irq)
		} else {
			panic!("Invalid interrupt type");
		};
		gic.set_interrupt_priority(uart_irqid, Some(cpu_id), 0x00)
			.unwrap();
		if (irqflags & 0xf) == 4 || (irqflags & 0xf) == 8 {
			gic.set_trigger(uart_irqid, Some(cpu_id), Trigger::Level)
				.unwrap();
		} else if (irqflags & 0xf) == 2 || (irqflags & 0xf) == 1 {
			gic.set_trigger(uart_irqid, Some(cpu_id), Trigger::Edge)
				.unwrap();
		} else {
			panic!("Invalid interrupt level!");
		}
		gic.enable_interrupt(uart_irqid, Some(cpu_id), true)
			.unwrap();
	}

	let reschedid = IntId::sgi(SGI_RESCHED.into());
	gic.set_interrupt_priority(reschedid, Some(cpu_id), 0x01)
		.unwrap();
	gic.enable_interrupt(reschedid, Some(cpu_id), true).unwrap();
	IRQ_NAMES.lock().insert(SGI_RESCHED, "Reschedule");

	*GIC.lock() = Some(gic);
}

// marks the given CPU core as awake
pub fn init_cpu() {
	let cpu_id: usize = core_id().try_into().unwrap();

	let mut gic = GIC.lock();
	let Some(gic) = &mut *gic else {
		return;
	};

	debug!("Mark cpu {cpu_id} as awake");

	gic.init_cpu(cpu_id);
	GicCpuInterface::enable_group1(true);
	GicCpuInterface::set_priority_mask(0xff);

	let fdt = env::fdt().unwrap();

	if let Some(timer_node) = fdt.find_compatible(&["arm,armv8-timer", "arm,armv7-timer"]) {
		let irq_slice = timer_node.property("interrupts").unwrap().value;
		/* Secure Phys IRQ */
		let (_irqtype, irq_slice) = irq_slice.split_at(size_of::<u32>());
		let (_irq, irq_slice) = irq_slice.split_at(size_of::<u32>());
		let (_irqflags, irq_slice) = irq_slice.split_at(size_of::<u32>());
		/* Non-secure Phys IRQ */
		let (irqtype, irq_slice) = irq_slice.split_at(size_of::<u32>());
		let (irq, irq_slice) = irq_slice.split_at(size_of::<u32>());
		let (irqflags, _irq_slice) = irq_slice.split_at(size_of::<u32>());
		let irqtype = u32::from_be_bytes(irqtype.try_into().unwrap());
		let irq = u32::from_be_bytes(irq.try_into().unwrap());
		let irqflags = u32::from_be_bytes(irqflags.try_into().unwrap());

		// enable timer interrupt
		let timer_irqid = if irqtype == 1 {
			IntId::ppi(irq)
		} else if irqtype == 0 {
			IntId::spi(irq)
		} else {
			panic!("Invalid interrupt type");
		};
		gic.set_interrupt_priority(timer_irqid, Some(cpu_id), 0x00)
			.unwrap();
		if (irqflags & 0xf) == 4 || (irqflags & 0xf) == 8 {
			gic.set_trigger(timer_irqid, Some(cpu_id), Trigger::Level)
				.unwrap();
		} else if (irqflags & 0xf) == 2 || (irqflags & 0xf) == 1 {
			gic.set_trigger(timer_irqid, Some(cpu_id), Trigger::Edge)
				.unwrap();
		} else {
			panic!("Invalid interrupt level!");
		}
		gic.enable_interrupt(timer_irqid, Some(cpu_id), true)
			.unwrap();
	}

	let reschedid = IntId::sgi(SGI_RESCHED.into());
	gic.set_interrupt_priority(reschedid, Some(cpu_id), 0x01)
		.unwrap();
	gic.enable_interrupt(reschedid, Some(cpu_id), true).unwrap();
}

static IRQ_NAMES: InterruptTicketMutex<HashMap<u8, &'static str, RandomState>> =
	InterruptTicketMutex::new(HashMap::with_hasher(RandomState::with_seeds(0, 0, 0, 0)));

#[allow(dead_code)]
pub(crate) fn add_irq_name(irq_number: u8, name: &'static str) {
	debug!("Register name \"{name}\" for interrupt {irq_number}");
	IRQ_NAMES.lock().insert(SPI_START + irq_number, name);
}

fn get_irq_name(irq_number: u8) -> Option<&'static str> {
	IRQ_NAMES.lock().get(&irq_number).copied()
}

pub(crate) static IRQ_COUNTERS: InterruptSpinMutex<BTreeMap<CoreId, &IrqStatistics>> =
	InterruptSpinMutex::new(BTreeMap::new());

pub(crate) struct IrqStatistics {
	pub counters: [AtomicU64; 256],
}

impl IrqStatistics {
	pub const fn new() -> Self {
		#[allow(clippy::declare_interior_mutable_const)]
		const NEW_COUNTER: AtomicU64 = AtomicU64::new(0);
		IrqStatistics {
			counters: [NEW_COUNTER; 256],
		}
	}

	pub fn inc(&self, pos: u8) {
		self.counters[usize::from(pos)].fetch_add(1, Ordering::Relaxed);
	}
}

pub(crate) fn print_statistics() {
	info!("Number of interrupts");
	for (core_id, irg_statistics) in IRQ_COUNTERS.lock().iter() {
		for (i, counter) in irg_statistics.counters.iter().enumerate() {
			let counter = counter.load(Ordering::Relaxed);
			if counter > 0 {
				match get_irq_name(i.try_into().unwrap()) {
					Some(name) => {
						info!("[{core_id}][{name}]: {counter}");
					}
					_ => {
						info!("[{core_id}][{i}]: {counter}");
					}
				}
			}
		}
	}
}
