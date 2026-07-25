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

use crate::arch::aarch64::kernel::core_local::{core_id, core_scheduler, increment_irq_counter, try_core_scheduler, CoreLocal};
use crate::arch::aarch64::kernel::scheduler::State;
use crate::scheduler::task::FrameLocation;
use crate::arch::aarch64::kernel::serial::handle_uart_interrupt;
use crate::arch::aarch64::mm::paging::{self, BasePageSize, PageSize, PageTableEntryFlags};
use crate::drivers::InterruptHandlerMap;
use crate::env;
use crate::mm::{PageAlloc, PageRangeAllocator};
use crate::scheduler::{self, CoreId, timer_interrupts};

/// R4-FU2 one-shot frame dump: task 1's known-good app panics with
/// `slice index 1610613287 (0x60000207)` (SPSR-shaped) — kernel-induced
/// corruption. The app handles its own panic in userspace and never calls
/// exit()/abort(), so the exit-path dump never fires. The only GUARANTEED
/// capture of task 1's live frame is at the exception entry (do_sync/do_irq),
/// which every syscall/IRQ hits. Dump the first N exceptions' full State.
/// `frame` is the `&State` trap_entry built (frame base), so it is exact
/// (no SP-prologue offset error).
static mut FRAME_DUMP_COUNT: u64 = 0;
const FRAME_DUMP_LIMIT: u64 = 6;

#[allow(dead_code)]
unsafe fn dump_frame_once(tag: &str, frame: *const State) {
	// Only capture task 1's frames (the faulting app task). The first
	// exceptions during boot belong to the idle task and would exhaust the
	// limit before task 1 runs.
	// Diagnostic path: may run pre-scheduler (early boot). Bail instead of
	// panicking via core_scheduler()'s unwrap.
	let Some(tid) = try_core_scheduler().map(|s| s.get_current_task_id()) else {
		return;
	};
	// Only capture task 1's frames (the faulting app task, "thread 'main' (1)").
	// `TaskId`'s inner field is private, so compare via its Display form.
	if format!("{}", tid) != "1" {
		return;
	}
	// R4-FU2: only capture task 1's USERSPACE (EL1t) exceptions, where
	// spsel==0x0. The EL1h (kernel-context) do_sync frames during network
	// init are repetitive noise that exhaust the dump limit before the
	// memory scan ever prints. The userspace frame is where 0x60000207
	// (SPSR-shaped) would surface in TLS/stack/heap.
	let base0 = frame as u64;
	let spsel0 = unsafe { core::ptr::addr_of!(*(base0 as *const u64)).read_volatile() };
	if spsel0 != 0x0 {
		return;
	}
	let c = unsafe { FRAME_DUMP_COUNT };
	if c >= FRAME_DUMP_LIMIT {
		return;
	}
	unsafe { FRAME_DUMP_COUNT = c + 1; }
	let base = frame as u64;
	let slot = base as *const u64;
	// Read key slots directly (State layout: spsel@0, elr@8, spsr@16,
	// sp_el0@24, tpidr@32, x0@40..x30@280).
	let spsel = unsafe { core::ptr::addr_of!(*slot.add(0)).read_volatile() };
	let elr = unsafe { core::ptr::addr_of!(*slot.add(1)).read_volatile() };
	let spsr = unsafe { core::ptr::addr_of!(*slot.add(2)).read_volatile() };
	let sp_el0 = unsafe { core::ptr::addr_of!(*slot.add(3)).read_volatile() };
	let tpidr = unsafe { core::ptr::addr_of!(*slot.add(4)).read_volatile() };
	error!(
		"[FRAME-DUMP #{c}] {tag} task={tid:?} frame_base={base:#x} | spsel={spsel:#x} elr={elr:#x} spsr={spsr:#x} sp_el0={sp_el0:#x} tpidr={tpidr:#x}"
	);
	// Dump all GPRs (x0..x30 = slots 5..35) inline.
	let mut hit_x = false;
	for i in 5..=35u64 {
		let v = unsafe { core::ptr::addr_of!(*slot.add(i as usize)).read_volatile() };
		if v == 0x60000207 {
			hit_x = true;
		}
		error!(
			"[FRAME-DUMP] x{}={:#x}{}",
			i - 5,
			v,
			if v == 0x60000207 { "  <<< MATCH" } else { "" }
		);
	}
	error!("[FRAME-DUMP] kernel_leak? x_slot={hit_x} (any x-slot == 0x60000207)",);
	// R4-FU2: kernel restore path is clean (no x-slot == 0x60000207), so the
	// bad value must live in task 1's USERSPACE memory. `0x60000207` is
	// SPSR-shaped (same family as task 1's saved SPSRs 0x60000204/0x60000205),
	// so an SPSR is leaking into a data word. Scan several task-1 regions for
	// it READ-ONLY. The two heap blocks below are stable across boots (seen in
	// ALLOC-TRACE: 0x800000018b00 and 0x80000002d160, size 0x12010 each).
	let scan = |name: &str, lo: u64, hi: u64| {
		let targets = [0x60000207u64, 0x7ffffffff8a0u64];
		let mut found: u64 = 0;
		let mut found_val: u64 = 0;
		for a in (lo..hi).step_by(8) {
			let v = unsafe { core::ptr::addr_of!(*(a as *const u64)).read_volatile() };
			if targets.contains(&v) {
				found = a;
				found_val = v;
				break;
			}
		}
		if found != 0 {
			error!("[FRAME-DUMP] SCAN[{name}] {found_val:#x} found at {found:#x}");
		} else {
			error!("[FRAME-DUMP] SCAN[{name}] neither target found in [{lo:#x}..{hi:#x})");
		}
	};
	// SAFE regions only (read_volatile faults on unmapped pages and halts):
	//  - TLS: ABOVE tpidr only. The page BELOW tpidr (tpidr-0x1000) is
	//    UNMAPPED (faults). Above tpidr, scan a SAFE 1 KiB window — TLS is
	//    typically <= 1 page, so 0x400 stays inside the mapped TLS block.
	//  - STACK: task 1's user stack is mapped [0x800015dd6000, 0x800015ed6000)
	//    (log: va=0x800015dd6000 size=0x100000). sp_el0 (~0x800015ed5ab0) is
	//    near the top. Clamp the LOW end to the stack base so we never read
	//    BELOW the mapping (sp-0x80000 would be 0x550 bytes under the base ->
	//    unmapped -> fault). Scan the full mapped stack up to sp.
	//  - HEAP0/HEAP1: the two stable blocks from ALLOC-TRACE.
	let sp = sp_el0;
	let stack_base: u64 = 0x800015dd6000;
	let stack_lo = core::cmp::max(sp.saturating_sub(0x80000), stack_base);
	scan("TLS", tpidr, tpidr + 0x400);
	scan("STACK", stack_lo, sp + 0x200);
	scan("HEAP0", 0x800000018b00, 0x800000018b00 + 0x12010);
	scan("HEAP1", 0x80000002d160, 0x80000002d160 + 0x12010);
}

/// R4-FU3: check the RESUME registers (what task 1 actually resumes with
/// after this exception — i.e. the frame `*state` as modified by the handler,
/// e.g. a syscall/page-fault return value written into x0). The entry-time
/// dump only shows ARGUMENTS; a corrupted SYSCALL RETURN in x0 would never
/// appear in memory (so the SCAN misses it) and never in the entry frame.
/// This fires at do_sync/do_irq EXIT (post-handler) and is SILENT unless it
/// finds 0x60000207 in an x-slot, so it cannot flood the log.
#[allow(dead_code)]
unsafe fn check_resume_x0(state: &State) {
	// Diagnostic path: may run pre-scheduler (early boot). Bail instead of
	// panicking via core_scheduler()'s unwrap.
	let Some(tid) = try_core_scheduler().map(|s| s.get_current_task_id()) else {
		return;
	};
	if format!("{}", tid) != "1" {
		return;
	}
	let slot = state as *const State as *const u64;
	let mut found = false;
	let mut vals = [0u64; 31];
	for i in 5..=35u64 {
		let v = unsafe { core::ptr::addr_of!(*slot.add(i as usize)).read_volatile() };
		vals[(i - 5) as usize] = v;
		if v == 0x60000207 {
			found = true;
		}
	}
	if found {
		error!("[RESUME-DUMP] task=1 found 0x60000207 in a resume GPR (syscall/page-fault return value)");
		for i in 0..31u64 {
			error!(
				"[RESUME-DUMP] x{}={:#x}{}",
				i,
				vals[i as usize],
				if vals[i as usize] == 0x60000207 { "  <<< MATCH" } else { "" }
			);
		}
	}
}

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
/// D4-TRACE (temporary, per-task-exception-slot-design R3-FU/nested): the
/// `trap_exit` D4 tail writes here on EVERY exception return so the fault
/// handler can prove/deny the nested-EL1h SP_EL1 clobber hypothesis without a
/// live debugger. Layout (u64 each):
///   [0] = saved SPSR_EL1 (M[3:0]: 0b0101=EL1h, 0b0100=EL1t) at this return
///   [1] = SP after the trap_exit ldp pops = the CORRECT SP to resume with
///         (for an EL1h return this is the interrupted context's live stack)
///   [2] = scratch_slot (CoreLocal@24) that D4 is about to force into SP_EL1
///   [3] = ELR_EL1 being returned to
///   [4] = monotonically incremented call counter (sanity/liveness)
///   [5] = magic 0xD4D4D4D4 once written at least once
#[unsafe(no_mangle)]
pub(crate) static mut D4_TRACE: [u64; 8] = [0u64; 8];
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
	unsafe { dump_frame_once("do_irq", _state as *const State) };
	unsafe { check_resume_x0(_state) };
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

	core_scheduler().scheduler(false).unwrap_or_default()
}

#[unsafe(no_mangle)]
pub(crate) extern "C" fn do_sync(state: &State, sp_el1: u64) {
	// NEW-1 (option-d-per-task-slot-rebased.md §10): for exceptions taken from
	// EL0t/EL1t (the interrupted context ran on SP_EL0), the D4 tail staged
	// SP_EL1 = the current task's scratch slot, so the trap frame must land there.
	// For exceptions taken from EL1h (the interrupted context ran ON SP_EL1 — a
	// kernel/handler stack, e.g. the C-runtime .init_array ctor walk), SP_EL1 is
	// the live kernel stack by design (same rule as the D4 tail's `tst x20,#1`
	// gate and el1_irq/el1_fiq). The slot invariant does NOT apply there.
	//   Gate on SPSR_EL1.M[0] (SPSEL bit): 0 => EL0t/EL1t, 1 => EL1h.
	// When the invariant applies, check BOTH:
	//   (a) ground truth — entry SP_EL1 (captured by `mov x1, sp` in el1_sync
	//       before trap_entry) equals the task's scratch_slot. If not, the frame
	//       landed on a shared/foreign stack (pool-select re-added? D4 tail
	//       bypassed?). This is the check that actually matters.
	//   (b) metadata — current task's frame_location == InSlot. Can lie under the
	//       ensure_slot pool-exhaustion fallback (§1.5): a task whose slot was NOT
	//       acquired keeps the default InSlot from Task::new() while its frame sits
	//       on the kernel stack. (a) catches that; (b) is consistency only.
	let from_el1h = (SPSR_EL1.get() & 1) == 1;
	if !from_el1h {
		let scratch = CoreLocal::get().scratch_slot();
		assert_eq!(
			sp_el1, scratch,
			"do_sync: entry SP_EL1 ({sp_el1:#x}) != CoreLocal.scratch_slot ({scratch:#x}) — frame not on task's slot (pool-select re-added? D4 tail bypassed?)"
		);
		assert_eq!(
			core_scheduler().get_current_task_frame_location(),
			FrameLocation::InSlot,
			"do_sync: current task frame_location must be InSlot"
		);
	}
	unsafe { dump_frame_once("do_sync", state as *const State) };
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
			error!("DIAG sp_el0_fault={sp_el0_fault:#x} far={far:#x} cur_task={cur_id:?}");
			error!("DIAG kernel_stack_top={ktop:#x} (base+KERNEL_STACK_SIZE)");
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

			// === D4-TRACE dump: the last trap_exit D4-tail decision inputs. ===
			// Proves/denies the nested-EL1h SP_EL1 clobber: if [0] M[3:0]==0b0101
			// (EL1h) AND [1] (the correct resume SP) != [2] (scratch_slot forced
			// into SP_EL1), then D4 destroyed an interrupted EL1h context's live
			// stack. [1] should be the stack the faulting ELR resumes on.
			{
				let t = unsafe { core::ptr::addr_of!(D4_TRACE).read_volatile() };
				let m = t[0] & 0xf;
				let el1h = m == 0b0101;
				error!(
					"D4-TRACE spsr={:#x} M[3:0]={m:#x} (EL1h={el1h}) resume_sp={:#x} scratch_slot={:#x} elr={:#x} calls={} magic={:#x}",
					t[0], t[1], t[2], t[3], t[4], t[5]
				);
				error!(
					"D4-TRACE clobbered_live_stack? {} (EL1h && resume_sp != scratch_slot)",
					el1h && t[1] != t[2]
				);
			}

			// === SLOT-REGION DIAGNOSTIC (per-task exception slot design) ===
			// If `far` lands inside the .exception_slots region, a slot
			// address leaked into task-visible state (a frame field or a
			// pointer the task followed at EL0). Dump the task's slot
			// assignment and the live frame's key fields to find which
			// field holds the bad slot pointer.
			{
				// Linker base of the slot section (single-core template;
				// matches protect_stack_guards' mapped region). Widen the
				// window to also catch UNDERFLOWS (far just BELOW the base),
				// which is the signature of the handler running with
				// SP_EL1 = slot base and growing downward past the slot.
				const SLOT_BASE: u64 = 0x4180e000;
				// stride = SLOT_SIZE + GUARD; total = SLOTS_PER_CORE * stride.
				const SLOT_STRIDE: u64 = (crate::config::EXCEPTION_SLOT_SIZE
					+ crate::config::EXCEPTION_SLOT_GUARD) as u64;
				const SLOT_END: u64 =
					SLOT_BASE + (crate::config::SLOTS_PER_CORE as u64) * SLOT_STRIDE;
				if far >= SLOT_BASE - SLOT_STRIDE && far < SLOT_END {
					// CoreLocal.scratch_slot is at offset 24; TPIDR_EL1 holds
					// &CoreLocal.
					let cl_ptr = TPIDR_EL1.get() as *const u64;
					let scratch_slot = unsafe { ptr::read_volatile(cl_ptr.add(24 / 8)) };
					error!(
						"DIAG-SLOT far={far:#x} (SLOT_BASE=0x4180e000); scratch_slot(CoreLocal@24)={scratch_slot:#x}"
					);
					// State is #[repr(C)] now (naturally aligned), so direct
					// field access is fine. `state` is a &State: read fields
					// straight through the reference.
					let f_spsr = state.spsr_el1;
					let f_elr = state.elr_el1 as *const () as u64;
					let f_sp_el0 = state.sp_el0;
					let f_tpidr = state.tpidr_el0;
					error!(
						"DIAG-SLOT frame: spsr={f_spsr:#x} elr={f_elr:#x} sp_el0={f_sp_el0:#x} tpidr={f_tpidr:#x}"
					);
					error!(
						"DIAG-SLOT frame: x0={0:#x} x1={1:#x} x2={2:#x} x3={3:#x} x4={4:#x} x5={5:#x} x6={6:#x} x7={7:#x}",
						state.x0,
						state.x1,
						state.x2,
						state.x3,
						state.x4,
						state.x5,
						state.x6,
						state.x7
					);
					error!(
						"DIAG-SLOT frame: x8={0:#x} x9={1:#x} x10={2:#x} x11={3:#x} x12={4:#x} x13={5:#x} x14={6:#x} x15={7:#x}",
						state.x8,
						state.x9,
						state.x10,
						state.x11,
						state.x12,
						state.x13,
						state.x14,
						state.x15
					);
					error!(
						"DIAG-SLOT frame: x16={0:#x} x17={1:#x} x18={2:#x} x19={3:#x} x20={4:#x} x21={5:#x} x22={6:#x} x23={7:#x}",
						state.x16,
						state.x17,
						state.x18,
						state.x19,
						state.x20,
						state.x21,
						state.x22,
						state.x23
					);
					error!(
						"DIAG-SLOT frame: x24={0:#x} x25={1:#x} x26={2:#x} x27={3:#x} x28={4:#x} x29={5:#x} x30={6:#x}",
						state.x24, state.x25, state.x26, state.x27, state.x28, state.x29, state.x30
					);
					// Also dump a window of the USER stack around sp_el0 so we
					// can see if the bad slot pointer was LOADED FROM the stack
					// (rather than sitting in a live GPR). Only read >= sp_el0
					// to avoid crossing a page boundary into an unmapped page
					// (which would fault inside the diagnostic itself). Guard
					// against a non-8-aligned sp_el0 (would UB in read_volatile).
					let usp = f_sp_el0;
					error!("DIAG-SLOT user_sp=0x{usp:x}");
					if usp != 0 && usp % 8 == 0 {
						for k in 0..16u64 {
							let va = usp + k * 8;
							let v = unsafe { ptr::read_volatile(va as *const u64) };
							error!("DIAG-SLOT ustk[{k}]@0x{va:x} = 0x{v:x}");
						}
					} else {
						error!("DIAG-SLOT user_sp not 8-aligned/zero; skipping ustk dump");
					}
					// Slot index + body/guard decode (signed so underflows show
					// as negative offsets, making the base-vs-top mistake obvious).
					let off = (far as i64) - (SLOT_BASE as i64);
					let underflow = off < 0;
					let stride = SLOT_STRIDE as i64;
					let body = crate::config::EXCEPTION_SLOT_SIZE as i64;
					let slot_idx = off / stride;
					let within = off % stride;
					let in_guard = within >= body;
					error!(
						"DIAG-SLOT far-off-from-base={off:#x} underflow={underflow} slot_idx={slot_idx} within_slot={within:#x} in_guard={in_guard} (stride={stride:#x}, body=[+0,+{body:#x}), guard=[+{body:#x},+{stride:#x}))"
					);
				}
			}

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
			state.x0,
			state.x1,
			state.x25,
			state.x30,
			state.spsel,
			state.spsr_el1,
			state.tpidr_el0,
			state.sp_el0,
		);
		let s_elr = state.elr_el1 as *const () as u64;
		error!(
			"[TRACE-SYNC] State @ {state_addr:#x}: elr_el1={s_elr:#x} x0={sx0:#x} x1={sx1:#x} x25={sx25:#x} x30={sx30_val:#x} spsel={sspsel:#x} spsr={sspsr:#x} tpidr={stpidr:#x} sp_el0={ssp_el0:#x}"
		);

		let task_id = core_scheduler().get_current_task_id();
		error!("Crashed in task {task_id:?}");

		// === INSTRUMENTATION: read raw State words for cross-check ===
		let raw = unsafe { core::slice::from_raw_parts(state as *const State as *const u64, 36) };
		error!(
			"[TRACE-SYNC] raw[0]={:#x} raw[1]={:#x} raw[8]={:#x} raw[35]={:#x}",
			raw[0], raw[1], raw[8], raw[35]
		);

		// === INSTRUMENTATION: check task_start trace buffer ===
		unsafe {
			let trace = crate::arch::aarch64::kernel::scheduler::TASK_START_TRACE;
			error!(
				"[TRACE-SYNC] TASK_START_TRACE[8]={:#x} (0x42=reached) [0]={:#x} [1]={:#x} [2]={:#x} [3]={:#x}",
				trace[8], trace[0], trace[1], trace[2], trace[3]
			);
			error!(
				"[TRACE-SYNC] TASK_START_TRACE SP_after_spsel={:#x} SP_after_func={:#x}",
				trace[4], trace[5]
			);
		}

		// === INSTRUMENTATION: dump stack memory around SP_EL0 and FP ===
		// SP_EL0 at fault time tells us where the function's stack was.
		// Dump from (SP_EL0 - 16) through (initial_SP + 16) to see the full frame.
		let sp_el0_fault = state.sp_el0;
		let fp_fault = sx29;
		if sp_el0_fault != 0 {
			// Use TASK_START_TRACE[4] as the initial SP_EL0 (set in task_start after spsel)
			let initial_sp_el0: u64 =
				unsafe { crate::arch::aarch64::kernel::scheduler::TASK_START_TRACE[4] };
			// Sentinel-encoding helper (mirror of create_stack_frame):
			// sentinel(w) = 0x5EED_0000_0000_0000 | (w & 0x0000_FFFF_FFFF_FFF8).
			let is_sentinel = |w: u64, addr: u64| -> bool {
				(w & 0xffff_0000_0000_0000) == 0x5eed_0000_0000_0000
					&& (w & 0x0000_ffff_ffff_ffff) == (addr & 0x0000_ffff_ffff_fff8)
			};
			// Classify: a window around SP_EL0. Report *how many* slots
			// broke and list the first/last broken VA so the footprint
			// and clobbered-slot is derivable from one line.
			let guard_page = if initial_sp_el0 > 0 {
				initial_sp_el0 + 16
			} else {
				sp_el0_fault + 0x200
			};
			let lo = sp_el0_fault.saturating_sub(16);
			let hi = guard_page.max(sp_el0_fault + 8);
			let nwords = ((hi - lo) / 8) as usize;
			if nwords > 0 && nwords <= 128 {
				let words = unsafe { core::slice::from_raw_parts(lo as *const u64, nwords) };
				error!(
					"[TRACE-SYNC] STACK DUMP lo={lo:#x} hi={hi:#x} ({} words):",
					nwords
				);
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
							if first_broke.is_none() {
								first_broke = Some(a);
							}
							last_broke = Some(a);
						}
					}
					error!(
						"  {:#x}: {:#x} {:#x} {:#x} {:#x}",
						addr, line[0], line[1], line[2], line[3]
					);
					i += show;
				}
				if let (Some(f), Some(l)) = (first_broke, last_broke) {
					error!(
						"[TRACE-SYNC] SENTINEL broken: count={} first_broke_va={:#x} last_broke_va={:#x} span={} bytes",
						broke_count,
						f,
						l,
						l - f + 8
					);
				} else {
					error!("[TRACE-SYNC] SENTINEL intact: no clobber in dumped window");
				}
			}
		}
		if fp_fault != 0 && fp_fault >= sp_el0_fault && fp_fault < 0x800015d92000 {
			let fp_words = unsafe { core::slice::from_raw_parts(fp_fault as *const u64, 4) };
			error!(
				"[TRACE-SYNC] Frame @ FP={fp_fault:#x}: [FP_chain={:#x} LR={:#x} {:#x} {:#x}]",
				fp_words[0], fp_words[1], fp_words[2], fp_words[3]
			);
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
		// R4-FU3: capture RESUME registers (post-handler) for task 1.
		unsafe { check_resume_x0(state); };
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
pub(crate) extern "C" fn do_error(_state: &State, sp_el1: u64) -> ! {
	// NEW-1 (option-d-per-task-slot-rebased.md §10): see do_sync. Only assert the
	// slot invariant for EL0t/EL1t-sourced exceptions; for EL1h SP_EL1 is the
	// live kernel stack by design (D4 tail SPSEL gate).
	if (SPSR_EL1.get() & 1) == 0 {
		let scratch = CoreLocal::get().scratch_slot();
		assert_eq!(
			sp_el1, scratch,
			"do_error: entry SP_EL1 ({sp_el1:#x}) != CoreLocal.scratch_slot ({scratch:#x}) — frame not on task's slot"
		);
		assert_eq!(
			core_scheduler().get_current_task_frame_location(),
			FrameLocation::InSlot,
			"do_error: current task frame_location must be InSlot"
		);
	}
	error!("Receive error interrupt");

	scheduler::abort()
}

/// Phase 4 double-fault handler (option-d-per-task-slot-rebased.md §5 / NEW-4).
///
/// Called from the vector-entry prologue (`df_fatal` in start.s) when the
/// double-fault predicate trips BEFORE `trap_entry` pushes a frame — i.e. the
/// interrupted context was already on the wrong/overflowing stack and pushing
/// 288 bytes there would corrupt it. The prologue has already:
///   - switched `sp` to the single `.overflow_stacks` rescue block (top),
///   - stored the original task `x0` in TPIDRRO_EL0 (restored into the frame),
///   - run `trap_entry`, so `state` is a normal trap frame on the rescue stack
///     (with `state.sp` = the ORIGINAL bad SP_EL1, captured as x1),
///   - set `x2 = flavor` (0=el1h_sync, 1=el1h_error, 2=el1t_sync, 3=el1t_error).
///
/// We print ESR/FAR/ELR + the bad SP + flavor, then fail-stop (spin/reset).
/// We do NOT attempt recovery — a nested/overflowing handler cannot be safely
/// unwound (avoids a recursive fault storm).
///
/// Compile-time cross-check: the immediates the asm depends on must match
/// config.rs. If someone changes a stack size without updating start.s, this
/// const-block fails at compile time (R4.4 — the corollary of NEW-4).
const _: () = {
	use crate::config::{EXCEPTION_SLOT_SIZE, EXCEPTION_SLOT_GUARD, KERNEL_STACK_SIZE};
	// df_check_el1h: slot_top - (EXCEPTION_SLOT_SIZE + EXCEPTION_SLOT_GUARD) = guard base
	assert!(
		(EXCEPTION_SLOT_SIZE + EXCEPTION_SLOT_GUARD) == 0x11000,
		"start.s df_check_el1h expects slot stride 0x11000"
	);
	// df_check_el1h danger window: GUARD + MARGIN (0x1000 + 0x3000 = 0x4000)
	assert!(
		(EXCEPTION_SLOT_GUARD + 0x3000) == 0x4000,
		"start.s df_check_el1h expects danger window 0x4000"
	);
	// df_fatal: .overflow_stacks stride == KERNEL_STACK_SIZE + GUARD(0x1000)
	assert!(
		(KERNEL_STACK_SIZE + 0x1000) == 0x9000,
		"start.s df_fatal expects .overflow_stacks stride 0x9000"
	);
	// df_fatal: rescue top offset == KERNEL_STACK_SIZE
	assert!(
		KERNEL_STACK_SIZE == 0x8000,
		"start.s df_fatal expects .overflow_stacks top at KERNEL_STACK_SIZE"
	);
};

#[unsafe(no_mangle)]
pub(crate) extern "C" fn do_double_fault(_state: &State, bad_sp: u64, flavor: u64) -> ! {
	let esr = ESR_EL1.get();
	let far = FAR_EL1.get();
	let elr = ELR_EL1.get();
	// DEFENSIVE: a double fault can fire before the per-core scheduler is
	// installed (early boot). `core_scheduler()` unwraps and would panic
	// BEFORE any diagnostics print — losing the entire dump. Use the
	// non-panicking accessor and degrade to task=None instead. (Found via the
	// Phase 5 injection harness: the boot-time udf injection panicked at
	// core_local.rs:245 with zero [DOUBLE-FAULT] output.)
	let tid = try_core_scheduler().map(|s| s.get_current_task_id());
	error!("============================================================");
	error!("[DOUBLE-FAULT] task={tid:?} flavor={flavor}");
	error!("  bad SP_EL1    = {bad_sp:#x}");
	error!("  ESR_EL1       = {esr:#x}");
	error!("  FAR_EL1       = {far:#x}");
	error!("  ELR_EL1       = {elr:#x}");
	error!("  slot top      = {:#x}", CoreLocal::get().scratch_slot());
	error!("  (exception taken while already on a slot / slot overflow)");
	error!("============================================================");
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
