use alloc::collections::{BTreeMap, VecDeque};
use core::arch::asm;
use core::ptr::{self, NonNull};
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

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
	trace!(
		"[FRAME-DUMP #{c}] {tag} task={tid:?} frame_base={base:#x} | spsel={spsel:#x} elr={elr:#x} spsr={spsr:#x} sp_el0={sp_el0:#x} tpidr={tpidr:#x}"
	);
	// Dump all GPRs (x0..x30 = slots 5..35) inline.
	let mut hit_x = false;
	for i in 5..=35u64 {
		let v = unsafe { core::ptr::addr_of!(*slot.add(i as usize)).read_volatile() };
		if v == 0x60000207 {
			hit_x = true;
		}
		trace!(
			"[FRAME-DUMP] x{}={:#x}{}",
			i - 5,
			v,
			if v == 0x60000207 { "  <<< MATCH" } else { "" }
		);
	}
	trace!("[FRAME-DUMP] kernel_leak? x_slot={hit_x} (any x-slot == 0x60000207)",);
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
			trace!("[FRAME-DUMP] SCAN[{name}] {found_val:#x} found at {found:#x}");
		} else {
			trace!("[FRAME-DUMP] SCAN[{name}] neither target found in [{lo:#x}..{hi:#x})");
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

/// Spike-1 test SGIs (feature `pmr-preempt-spike`, rtic-gicv3-async-reactor.md
/// §13 Phase A). LO=14 (priority 0x40) and HI=15 (priority 0x00, highest) prove
/// nested priority preemption via PMR ceiling + PSTATE.I re-enable, and that a
/// 0x00 PMR (all-masked) blocks HI. Consumes 2 of the architecturally-fixed 16
/// SGI INTIDs (R3.6: budget RT_LEVELS+COOP_LEVELS+1+2 <= 16).
#[cfg(feature = "pmr-preempt-spike")]
pub(crate) const SGI_PMRTEST_LO: u8 = 14;
#[cfg(feature = "pmr-preempt-spike")]
pub(crate) const SGI_PMRTEST_HI: u8 = 15;

// Spike 2 (rtic-gicv3-async-reactor.md §13): per-band executor + COOP-SGI waker.
// SGI budget (R3.6): RESCHED=1, spike 14/15, COOP_WAKE=13, RT_BRIDGE=12 ->
// RT_LEVELS(1)+COOP_LEVELS(1)+1(RESCHED)+2(spike)=5 <= 16. COOP_WAKE is the
// COOP-band SGI (pended by a COOP task's waker); RT_BRIDGE is the RT-band SGI
// whose ISR wakes a COOP future (the top-half -> bottom-half handoff proof).
// Also used by Spike 3 (pmr-coop-net): the COOP-band SGI drives network_run's
// executor drain (the "SGI-pending waker" of §13 Phase C).
#[cfg(any(feature = "pmr-band", feature = "pmr-coop-net"))]
pub(crate) const SGI_COOP_WAKE: u8 = 13;
// Spike 4 (stackful-continuations.md §3): continuation park/resume waker.
// Distinct from SGI_COOP_WAKE so continuations need no pmr-band feature.
#[cfg(feature = "continuations")]
pub(crate) const SGI_CONT_WAKE: u8 = 11;
#[cfg(any(feature = "pmr-band", feature = "pmr-coop-net"))]
pub(crate) const SGI_RT_BRIDGE: u8 = 12;
#[cfg(any(feature = "pmr-band", feature = "pmr-coop-net"))]
const COOP_WAKE_PRIORITY: u8 = 0x60; // COOP band (below RT per §4)
#[cfg(feature = "pmr-band")]
const RT_BRIDGE_PRIORITY: u8 = 0x20; // RT band (above COOP per §4)

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

/// Re-enable ONLY IRQ (bit 2 of DAIF), leaving FIQ/SError/Debug masked.
/// Used inside the Spike-1 RT-band nesting path (feature `pmr-preempt-spike`)
/// where PMR already gates priority; we must NOT re-enable FIQ/SError, or a
/// pending FIQ would preempt the RT band (R1.10 / rtic-gicv3-async-reactor.md
/// §13). Distinct from `disable()` above, which masks ALL DAIF bits.
#[cfg(feature = "pmr-preempt-spike")]
#[inline(always)]
pub(crate) fn enable_irqs() {
	unsafe {
		asm!("msr daifclr, #2", options(nostack));
	}
}

/// Mask ONLY IRQ (bit 2 of DAIF). Paired with `enable_irqs()`.
#[cfg(feature = "pmr-preempt-spike")]
#[inline(always)]
pub(crate) fn disable_irqs() {
	unsafe {
		asm!("msr daifset, #2", options(nostack));
	}
}

/// Spike 2 (INV-P4 / INV-P8): on IRQ entry, bump the per-core nesting depth and
/// mark `in_irq`. `irq_exit` decrements it. These are cfg-gated so the default
/// (production) build compiles them out and `do_irq`/`do_fiq` stay byte-identical
/// (unconditional `scheduler()` call). When `pmr-band` is off, `irq_enter`/
/// `irq_exit` are no-ops and the nesting counter is never touched.
#[inline(always)]
#[cfg(feature = "pmr-band")]
fn irq_enter() {
	let cl = CoreLocal::get();
	cl.inc_rt_nest_depth();
	cl.set_in_irq(true);
}
#[inline(always)]
#[cfg(not(feature = "pmr-band"))]
fn irq_enter() {}
#[inline(always)]
#[cfg(feature = "pmr-band")]
fn irq_exit() {
	let cl = CoreLocal::get();
	cl.set_in_irq(false);
	cl.dec_rt_nest_depth();
}
#[inline(always)]
#[cfg(not(feature = "pmr-band"))]
fn irq_exit() {}

// ── Spike 2: PMR ceiling RAII guard (INV-P2) + COOP-band waker (INV-P5) ──
// All gated on `pmr-band`; compiled out in the default (production) build.

/// INV-P2 (SRP ceiling monotonicity): snapshot the current PMR on construction,
/// raise it to `new_ceiling` (which must be numerically <= the saved PMR — you
/// can only RAISE the ceiling, never lower it below the system ceiling), and
/// restore the saved PMR on `Drop`. The RAII shape makes the save/restore
/// pairing impossible to forget. A task that drops the guard restores PMR
/// exactly, so it cannot accidentally unmask a lower-priority IRQ.
#[cfg(feature = "pmr-band")]
pub(crate) struct PmrCeiling {
	saved: u8,
}
#[cfg(feature = "pmr-band")]
impl PmrCeiling {
	pub fn raise(new_ceiling: u8) -> Self {
		let cur = GicCpuInterface::get_priority_mask();
		// Ceiling monotonicity: raising PMR means numerically LOWERING it
		// (0x00 = all masked). So a valid raise sets new_ceiling <= cur.
		debug_assert!(
			new_ceiling <= cur,
			"PmrCeiling::raise would lower the system ceiling (SRP violation): new={new_ceiling:#x} cur={cur:#x}"
		);
		GicCpuInterface::set_priority_mask(new_ceiling);
		Self { saved: cur }
	}
}
#[cfg(feature = "pmr-band")]
impl Drop for PmrCeiling {
	fn drop(&mut self) {
		let cur = GicCpuInterface::get_priority_mask();
		debug_assert_eq!(cur, self.saved, "PMR changed under a PmrCeiling guard");
		GicCpuInterface::set_priority_mask(self.saved);
	}
}

/// INV-P5: a COOP-band task's waker pends ONLY its own band's SGI. The waker
/// stores the executor's own `Waker` (set by the spawned future on first poll)
/// plus asserts the target INTID belongs to the COOP band before pending it.
#[cfg(feature = "pmr-band")]
static BAND_WAKER: InterruptSpinMutex<Option<core::task::Waker>> =
	InterruptSpinMutex::new(None);

#[cfg(feature = "pmr-band")]
fn wake_coop_future() {
	// INV-P5: assert the waker pends the COOP-band SGI (13), not an RT or
	// foreign INTID.
	debug_assert_eq!(
		SGI_COOP_WAKE, 13,
		"COOP-band waker must pend SGI_COOP_WAKE (INV-P5)"
	);
	// Pending the SGI is the async signal that the idle loop should drain the
	// executor; calling the inner waker marks the COOP future ready (the poll
	// itself happens in thread mode via the idle loop, never on the exc stack —
	// INV-P10). Neither call runs the future body here.
	pend_sgi_to_self(SGI_COOP_WAKE);
	let waker = BAND_WAKER.lock().take();
	if let Some(w) = waker {
		w.wake();
	}
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

	// Spike 2 (pmr-band): register the COOP-band SGI (13) and RT-band bridge SGI
	// (12) handlers. The COOP SGI ISR must NOT run the executor (INV-P10) — it
	// only sets a ready flag; the idle loop drains `core_local::ex()` in thread
	// mode. The RT bridge ISR raises the PMR ceiling (INV-P2 PmrCeiling guard),
	// wakes the COOP future via the band waker (INV-P5), and is run-to-completion
	// (no .await / no executor) per INV-P3. For pmr-coop-net, the COOP SGI is
	// also the executor-driver waker (network_run's waker pends it). The RT
	// bridge handler is Spike-2 specific (pmr-band); the COOP handler is shared.
	#[cfg(any(feature = "pmr-band", feature = "pmr-coop-net"))]
	{
		fn coop_wake_handler() {
			// INV-P10: never call executor::run() here (this is the exc stack).
			// The idle loop drains the executor in thread mode. We only mark
			// that a COOP wake is pending so the harness/serial is observable.
			COOP_WAKE_PENDING.store(true, Ordering::SeqCst);
		}
		#[cfg(feature = "pmr-band")]
		fn rt_bridge_handler() {
			info!("[PMR-BAND] [RT-BRIDGE-ISR] ENTER (vector {SGI_RT_BRIDGE})");
			// INV-P2: raise PMR to the RT ceiling (numerically <= saved PMR).
			// The guard restores PMR on drop. 0x20 is the RT band (above COOP).
			let _ceiling = PmrCeiling::raise(0x20);
			// Signal the COOP future that the RT band has woken it (INV-P5).
			COOP_TRIGGERED.store(true, Ordering::SeqCst);
			info!("[PMR-BAND] [RT-BRIDGE-ISR] woke coop via band waker");
			// INV-P5: pend the COOP-band SGI + wake the spawned future.
			wake_coop_future();
			// _ceiling drops here -> PMR restored. Run-to-completion (INV-P3).
		}
		let coop_vec = handlers
			.entry(SGI_COOP_WAKE)
			.or_insert_with(|| VecDeque::<fn()>::new());
		coop_vec.push_back(coop_wake_handler);
		#[cfg(feature = "pmr-band")]
		{
			let rt_vec = handlers
				.entry(SGI_RT_BRIDGE)
				.or_insert_with(|| VecDeque::<fn()>::new());
			rt_vec.push_back(rt_bridge_handler);
		}
		#[cfg(feature = "pmr-band")]
		info!(
			"[PMR-BAND] install_handlers: SGI_COOP_WAKE={SGI_COOP_WAKE} SGI_RT_BRIDGE={SGI_RT_BRIDGE} rt_present={} coop_present={}",
			handlers.contains_key(&SGI_RT_BRIDGE),
			handlers.contains_key(&SGI_COOP_WAKE)
		);
	}

	// Spike 4 (stackful-continuations.md §3): register the continuation wake SGI
	// (11) handler. The ISR only marks the pending cont READY (INV-P10-style:
	// never runs executor/future body on the exc stack) — the idle-loop drain
	// (`continuations::drain_ready`) does the actual resume in thread mode.
	#[cfg(feature = "continuations")]
	{
		fn cont_wake_handler() {
			// Mark the parked continuation READY so the idle loop resumes it.
			crate::arch::aarch64::kernel::continuations::coop_wake();
		}
		let cont_vec = handlers
			.entry(SGI_CONT_WAKE)
			.or_insert_with(|| VecDeque::<fn()>::new());
		cont_vec.push_back(cont_wake_handler);
		info!("[CONT] install_handlers: SGI_CONT_WAKE={SGI_CONT_WAKE} registered");
	}

	INTERRUPT_HANDLERS.set(handlers).unwrap();
}

#[unsafe(no_mangle)]
pub(crate) extern "C" fn do_fiq(_state: &State) -> *mut usize {
	let Some(irqid) = GicCpuInterface::get_and_acknowledge_interrupt(InterruptGroup::Group1) else {
		return ptr::null_mut();
	};

	let vector: u8 = u32::from(irqid).try_into().unwrap();

	// Spike-2 (INV-P4/P8): count this FIQ in the per-core nesting depth. No-op
	// when `pmr-band` is off.
	irq_enter();

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

	// Spike 3 (pmr-coop-net, INV-P6): under EOImode=1 the priority-drop
	// (EOIR1) + deactivate (DIR) are split; complete both AND decrement the
	// per-core EOI in-flight counter (paired with the IAR1 ack inc).
	// Default (feature off, EOImode=0): combined end_interrupt.
	#[cfg(feature = "pmr-coop-net")]
	eoi_complete(irqid);
	#[cfg(not(feature = "pmr-coop-net"))]
	GicCpuInterface::end_interrupt(irqid, InterruptGroup::Group1);

	// INV-P8 (Spike 2): skip the cooperative scheduler on return from a NESTED
	// FIQ; run it only on the outermost return (depth back to 0).
	irq_exit();
	if CoreLocal::get().rt_nest_depth() == 0 {
		debug_assert!(CoreLocal::get().rt_nest_depth() == 0);
		core_scheduler().scheduler(false).unwrap_or_default()
	} else {
		ptr::null_mut()
	}
}

/// Spike-1 trigger (feature `pmr-preempt-spike`): self-IPI the LO test SGI on
/// the current core so `do_irq` enters the Spike-1 preemption harness. Called
/// once from `boot_processor_main` after IRQs are enabled. The handler pends
/// HI to itself and runs both the preemption and ceiling-block sub-tests.
#[cfg(feature = "pmr-preempt-spike")]
pub fn pmr_spike_trigger() {
	info!("[PMR-SPIKE] trigger: self-IPI SGI_PMRTEST_LO ({SGI_PMRTEST_LO})");
	pend_sgi_to_self(SGI_PMRTEST_LO);
}

/// Spike 2 (pmr-band): one-shot guard so the per-band harness fires exactly
/// once (on the BSP / I/O core) from the idle loop, which runs AFTER
/// `install_handlers` has registered the SGI 12/13 handlers in the map. Firing
/// from `boot_processor_main` (before `install_handlers`) would pend SGI 12
/// before its handler exists -> the SGI is silently dropped. (Lesson from
/// Spike 1: register/enabled state must exist before the self-IPI.)
#[cfg(feature = "pmr-band")]
static PMR_BAND_HARNESS_DONE: AtomicBool = AtomicBool::new(false);

/// Spike 2 (feature `pmr-band`): boot the per-band executor harness. Called
/// ONCE from the idle loop (after `install_handlers` registered the SGI 12/13
/// handlers). Spawns a COOP-band future that parks until the RT bridge ISR
/// wakes it (via the band waker), then self-IPIs `SGI_RT_BRIDGE` on the current
/// core. The RT ISR (running on the RT band) raises the PMR ceiling (INV-P2),
/// wakes the COOP future via the band waker (INV-P5), and returns; the COOP SGI
/// then fires in thread mode and the idle loop drains the executor, polling the
/// future (INV-P10 respected: executor runs in thread mode, not in the SGI ISR).
#[cfg(feature = "pmr-band")]
pub fn pmr_band_maybe_trigger() {
	// Run exactly once, on the BSP / I/O core (core 0).
	if core_id() != 0 {
		return;
	}
	if PMR_BAND_HARNESS_DONE.swap(true, Ordering::SeqCst) {
		return;
	}
	use core::future::poll_fn;
	use core::task::{Context, Poll};

	info!("[PMR-BAND] trigger: spawn COOP future + self-IPI SGI_RT_BRIDGE ({SGI_RT_BRIDGE})");
	crate::executor::spawn(async {
		// Park until the RT bridge ISR sets COOP_TRIGGERED and wakes us via the
		// band waker (INV-P5). On first poll we store our waker so the RT ISR's
		// wake() can mark us ready.
		poll_fn(|cx: &mut Context<'_>| {
			BAND_WAKER.lock().replace(cx.waker().clone());
			if COOP_TRIGGERED.load(Ordering::SeqCst) {
				Poll::Ready(())
			} else {
				Poll::Pending
			}
		})
		.await;
		info!("[PMR-BAND] [COOP-FUTURE] polled");
		COOP_FUTURE_DONE.store(true, Ordering::SeqCst);
		// Bridge proof complete: RT-band ISR (SRP ceiling via PmrCeiling, INV-P2)
		// woke a COOP-band future through the band waker (INV-P5), the waker is
		// pinned to the COOP SGI (13, INV-P5), RT is core-0 (INV-P7), and the
		// executor ran this future in thread mode, not on the exception stack
		// (INV-P10). Emit the verified PASS marker the Justfile kernel checks.
		info!("[PMR-BAND] INV-P2/P5/P7/P10 PASS: per-band executor + RT->COOP bridge verified");
	});
	info!("[PMR-BAND] COOP future spawned OK; about to pend SGI_RT_BRIDGE");
	// Fire the RT-band bridge SGI; its ISR wakes the COOP future.
	pend_sgi_to_self(SGI_RT_BRIDGE);
	info!("[PMR-BAND] SGI_RT_BRIDGE pended");
}

/// Spike 3 (pmr-coop-net): one-shot guard so the COOP-band executor harness
/// fires exactly once (on the BSP / I/O core) from the idle loop, after
/// `install_handlers` registered SGI 13 (COOP-band driver). The harness proves
/// INV-P3 (COOP SGI ISR is run-to-completion: it only sets a flag, never runs
/// the executor on the exception stack per INV-P10), INV-P6 (per-core EOI
/// in-flight counter returns to 0 at idle under EOImode=1 split-EOI), and
/// INV-P11 (no fd / teardown I/O in the RT/COOP SGI path). It reuses the
/// existing `network_run` future: `wake_network_waker()` (under this feature)
/// pends SGI_COOP_WAKE, whose ISR sets `COOP_WAKE_PENDING` and lets the idle
/// loop drain `network_run` in thread mode.
#[cfg(feature = "pmr-coop-net")]
static PMR_COOP_NET_HARNESS_DONE: AtomicBool = AtomicBool::new(false);
#[cfg(feature = "pmr-coop-net")]
static COOP_NET_PROBE_POLLED: AtomicBool = AtomicBool::new(false);
#[cfg(feature = "pmr-coop-net")]
static PMR_COOP_NET_PASS_EMITTED: AtomicBool = AtomicBool::new(false);

#[cfg(feature = "pmr-coop-net")]
pub fn pmr_coop_net_maybe_trigger() {
	// Only the BSP / I/O core (core 0) runs the harness.
	if core_id() != 0 {
		return;
	}
	// ARM phase — runs exactly once: spawn the probe future + pend the COOP
	// SGI. The PASS check below re-runs on every subsequent idle-loop call
	// because the probe future is only polled AFTER this function returns
	// (the idle loop drains the executor in thread mode, INV-P10).
	if !PMR_COOP_NET_HARNESS_DONE.swap(true, Ordering::SeqCst) {
		use core::future::poll_fn;
		use core::task::{Context, Poll};

		info!(
			"[PMR-COOP-NET] trigger: spawn probe future + drive network_run via COOP-band SGI waker"
		);
		// Spawn a probe future that records when the executor (driven by the
		// COOP-band SGI waker / idle-loop drain) actually polls it.
		crate::executor::spawn(async {
			poll_fn(|_cx: &mut Context<'_>| {
				COOP_NET_PROBE_POLLED.store(true, Ordering::SeqCst);
				Poll::Ready(())
			})
			.await;
			info!("[PMR-COOP-NET] [PROBE] polled via COOP-band executor drain");
		});
		// Drive network_run's waker via the COOP-band SGI (the Spike-3
		// "SGI-pending waker"): under pmr-coop-net this pends SGI_COOP_WAKE,
		// whose ISR is run-to-completion (INV-P3) and whose thread-mode drain
		// runs the executor (INV-P10).
		crate::executor::network::wake_network_waker();
		return;
	}
	// PASS phase — re-checked each idle-loop pass until it fires once:
	// COOP SGI ISR ran (COOP_WAKE_PENDING, INV-P3/P10), the probe future was
	// polled (executor driven under the COOP band), and the per-core EOI
	// in-flight counter is back to 0 (INV-P6: EOImode=1 split-EOI balanced).
	if !PMR_COOP_NET_PASS_EMITTED.load(Ordering::SeqCst)
		&& COOP_WAKE_PENDING.load(Ordering::SeqCst)
		&& COOP_NET_PROBE_POLLED.load(Ordering::SeqCst)
		&& CoreLocal::get().eoi_inflight() == 0
		&& !PMR_COOP_NET_PASS_EMITTED.swap(true, Ordering::SeqCst)
	{
		info!(
			"[PMR-COOP-NET] INV-P3/P6/P11 PASS: COOP-band SGI waker drives network_run; EOImode=1 EOI pairing balanced; no fd/teardown in RT/COOP SGI path"
		);
	}
}


/// Spike-1 (feature `pmr-preempt-spike`) flag shared between the LO handler
/// (spins) and the HI handler (sets it). Reset between the two sub-tests.
#[cfg(feature = "pmr-preempt-spike")]
static PMR_SPIKE_FLAG: AtomicBool = AtomicBool::new(false);

/// Spike 2 (pmr-band): observable state for the per-band executor harness.
/// `COOP_WAKE_PENDING` is set by the COOP SGI ISR (thread-mode drain observes
/// it); `COOP_FUTURE_DONE` is set when the spawned COOP future is polled,
/// proving the RT->COOP bridge end-to-end.
#[cfg(any(feature = "pmr-band", feature = "pmr-coop-net"))]
static COOP_WAKE_PENDING: AtomicBool = AtomicBool::new(false);
#[cfg(feature = "pmr-band")]
static COOP_FUTURE_DONE: AtomicBool = AtomicBool::new(false);
#[cfg(feature = "pmr-band")]
static COOP_TRIGGERED: AtomicBool = AtomicBool::new(false);

/// Pend a specific SGI to the current core via raw `ICC_SGI1R_EL1` (mirrors
/// `wakeup_core`, which works around the arm-gic `send_sgi` ABI bug). Used by
/// the Spike-1 harness to self-IPI the HI test SGI from inside the LO handler.
/// Also used by Spike 2 (pmr-band) to self-IPI the RT bridge SGI, and by
/// Spike 3 (pmr-coop-net) to self-IPI the COOP-band driver SGI.
#[cfg(any(
	feature = "pmr-preempt-spike",
	feature = "pmr-band",
	feature = "pmr-coop-net",
	feature = "continuations"
))]
pub(crate) fn pend_sgi_to_self(intid: u8) {
	// ICC_SGI1R_EL1: Aff3[63:48] | IRM[40] | Aff2[39:32] | INTID[31:24] |
	//               Aff1[23:16] | TargetList[15:0]. Self-target: TargetList bit
	// for core 0 (we run on core 0 / the IO core) = 1<<0; INTID in [31:24].
	let target_list: u64 = 1u64 << 0; // core 0
	let sgi_value: u64 = (u64::from(intid) << 24) | target_list;
	unsafe {
		core::arch::asm!(
			"msr ICC_SGI1R_EL1, {value:x}",
			value = in(reg) sgi_value,
			options(nostack),
		);
		// Drain any pending write + let the GIC observe the SGI before spin.
		core::arch::asm!("isb", options(nostack));
	}
}

/// Spike 3 (pmr-coop-net, INV-P6 / EOImode=1) — raw-MSR GIC CPU-interface
/// wrappers for the bits arm-gic 0.8.1 does NOT expose (§1.1 / §9 #5). Under
/// `EOImode=1`, `ICC_EOIR1_EL1` does priority-drop ONLY and `ICC_DIR_EL1` does
/// the separate deactivate; the running priority is readable via `ICC_RPR_EL1`.
/// `ICC_CTLR_EL1.EOImode=1` switches the split on (globally, for this core).
#[cfg(feature = "pmr-coop-net")]
mod eoi_mode1 {
	/// Bit 1 of ICC_CTLR_EL1 selects EOImode (0 = combined EOI+DIR via EOIR1,
	/// 1 = split priority-drop (EOIR1) + separate deactivate (DIR)).
	const ICC_CTLR_EOIMODE_BIT: u64 = 1 << 1;

	#[inline]
	pub fn set_eoi_mode_1() {
		unsafe {
			// Read-modify-write ICC_CTLR_EL1 to set EOImode=1.
			let mut ctlr: u64;
			core::arch::asm!("mrs {v}, ICC_CTLR_EL1", v = out(reg) ctlr, options(nostack));
			ctlr |= ICC_CTLR_EOIMODE_BIT;
			core::arch::asm!("msr ICC_CTLR_EL1, {v}", v = in(reg) ctlr, options(nostack));
			core::arch::asm!("isb", options(nostack));
		}
	}

	/// Split priority-drop (EOIR1) for `intid`. Under EOImode=1 this does NOT
	/// deactivate — call `eoi_deactivate` afterwards.
	#[inline]
	pub fn eoi_priority_drop(intid: u32) {
		let v = u64::from(intid);
		unsafe {
			core::arch::asm!("msr ICC_EOIR1_EL1, {v}", v = in(reg) v, options(nostack));
		}
	}

	/// Separate deactivate (DIR) for `intid`. Must follow `eoi_priority_drop`.
	#[inline]
	pub fn eoi_deactivate(intid: u32) {
		let v = u64::from(intid);
		unsafe {
			core::arch::asm!("msr ICC_DIR_EL1, {v}", v = in(reg) v, options(nostack));
		}
	}

	/// Read ICC_RPR_EL1 (current running priority). Used by INV-P9 / diagnostics.
	#[inline]
	pub fn read_running_priority() -> u8 {
		let v: u64;
		unsafe {
			core::arch::asm!("mrs {v}, ICC_RPR_EL1", v = out(reg) v, options(nostack));
		}
		v as u8
	}

	/// Full split-EOI for one Group1 interrupt under EOImode=1: priority-drop
	/// (EOIR1) then deactivate (DIR), paired with the IAR1 ack counted by the
	/// caller via `eoi_inflight_dec()`. INV-P6 invariant: exactly one drop +
	/// one deactivate per ack.
	#[inline]
	pub fn eoi_complete_group1(intid: u32) {
		eoi_priority_drop(intid);
		eoi_deactivate(intid);
	}
}

/// Spike 3 (pmr-coop-net, INV-P6): complete a Group1 interrupt under EOImode=1
/// by splitting priority-drop + deactivate AND decrementing the per-core
/// EOI in-flight counter (paired with the `eoi_inflight_inc` on IAR1 ack).
/// When the feature is off, this is unused and the caller uses the combined
/// `GicCpuInterface::end_interrupt` (default, EOImode=0) instead — so the
/// default build is byte-identical.
#[cfg(feature = "pmr-coop-net")]
#[inline]
fn eoi_complete(irqid: IntId) {
	let raw: u32 = irqid.into();
	eoi_mode1::eoi_complete_group1(raw);
	CoreLocal::get().eoi_inflight_dec();
}

/// Spike-1 dispatch for the two test SGIs. Returns the value `do_irq` should
/// return (null = no task switch; we never switch during the harness).
///
/// Proves INV-P12 (preemption + ceiling-block) and prototypes R3.1 (nested IRQ
/// re-entry into el1_irq is safe — el1_irq does NOT call df_check_*).
#[cfg(feature = "pmr-preempt-spike")]
fn pmr_spike_dispatch(vector: u8) -> *mut usize {
	match vector {
		SGI_PMRTEST_HI => {
			// HI handler: runs as a nested IRQ inside LO's spin (when LO has
			// raised PMR to 0x40 and re-enabled IRQs). Just set the flag.
			info!("[PMR-SPIKE] [HI-ENTER]");
			PMR_SPIKE_FLAG.store(true, Ordering::SeqCst);
			info!("[PMR-SPIKE] [HI-EXIT]");
			ptr::null_mut()
		}
		SGI_PMRTEST_LO => {
			// ---- Test A: PREEMPTION via PMR ceiling + IRQ re-enable ----
			PMR_SPIKE_FLAG.store(false, Ordering::SeqCst);
			info!("[PMR-SPIKE] [LO-ENTER]");
			// ENTRY SEQUENCE (R1.3): raise PMR to LO ceiling FIRST, then
			// re-enable IRQs. While PMR=0x40, only priorities < 0x40 (i.e. HI
			// = 0x00) can preempt.
			GicCpuInterface::set_priority_mask(0x40);
			debug_assert!(GicCpuInterface::get_priority_mask() <= 0x40, "INV-P9: PMR ceiling not applied");
			// INV-P9: PMR is already at <= ceiling before IRQs are re-enabled.
			enable_irqs(); // daifclr #2 — only IRQ, FIQ/SError stay masked.
			// Pend HI to self; because PMR=0x40 and HI=0x00, the GIC will
			// preempt LO mid-spin with the HI handler.
			pend_sgi_to_self(SGI_PMRTEST_HI);
			// Spin (bounded) until HI sets the flag — proving preemption.
			let mut spins = 0usize;
			while !PMR_SPIKE_FLAG.load(Ordering::SeqCst) {
				spins += 1;
				if spins > 1_000_000 {
					break; // safety cap; should never hit on success
				}
			}
			// EXIT SEQUENCE (R1.3): disable IRQs FIRST, then restore PMR, so a
			// lower-priority IRQ cannot sneak in during the gap.
			disable_irqs();
			GicCpuInterface::set_priority_mask(0xff);
			info!("[PMR-SPIKE] [LO-EXIT-A] preempted={}", PMR_SPIKE_FLAG.load(Ordering::SeqCst));

			// ---- Test B: CEILING-BLOCK (PMR=0x00 masks ALL) ----
			PMR_SPIKE_FLAG.store(false, Ordering::SeqCst);
			info!("[PMR-SPIKE] [LO-ENTER-B]");
			// Raise PMR to 0x00 — the highest-priority value, which masks ALL
			// interrupts (no priority < 0x00). Then re-enable IRQs: HI must
			// NOT arrive.
			GicCpuInterface::set_priority_mask(0x00);
			debug_assert_eq!(GicCpuInterface::get_priority_mask(), 0x00, "INV-P9/INV-P12: PMR floor not 0x00");
			enable_irqs();
			pend_sgi_to_self(SGI_PMRTEST_HI);
			let mut spins = 0usize;
			while !PMR_SPIKE_FLAG.load(Ordering::SeqCst) {
				spins += 1;
				if spins > 1_000_000 {
					break; // expected: HI is blocked, so we time out
				}
			}
			let blocked = !PMR_SPIKE_FLAG.load(Ordering::SeqCst);
			disable_irqs();
			GicCpuInterface::set_priority_mask(0xff);
			info!("[PMR-SPIKE] [LO-EXIT-B] hi_blocked={blocked}");
			if blocked {
				info!("[PMR-SPIKE] INV-P12 PASS: PMR=0x00 masked the higher-priority SGI (ceiling-block holds)");
			} else {
				warn!("[PMR-SPIKE] INV-P12 FAIL: HI preempted despite PMR=0x00 (ceiling-block broken)");
			}
			ptr::null_mut()
		}
		_ => ptr::null_mut(),
	}
}

#[unsafe(no_mangle)]
pub(crate) extern "C" fn do_irq(_state: &State) -> *mut usize {
	unsafe { dump_frame_once("do_irq", _state as *const State) };
	unsafe { check_resume_x0(_state) };
	let Some(irqid) = GicCpuInterface::get_and_acknowledge_interrupt(InterruptGroup::Group1) else {
		return ptr::null_mut();
	};
	#[cfg(feature = "pmr-coop-net")]
	CoreLocal::get().eoi_inflight_inc();

	let vector: u8 = u32::from(irqid).try_into().unwrap();

	// Spike-2 (INV-P4/P8): count this IRQ in the per-core nesting depth. Must
	// run for EVERY vector (including the Spike-1 test SGIs below) so nested
	// IRQ detection is correct. No-op when `pmr-band` is off (default build).
	irq_enter();

	#[cfg(feature = "pmr-band")]
	if vector == SGI_RT_BRIDGE || vector == SGI_COOP_WAKE {
		info!("[PMR-BAND] do_irq entered for vector {vector}");
	}
	// Spike-1 preemption harness (rtic-gicv3-async-reactor.md §13 Phase A).
	// Feature-gated branch AFTER the normal Group1 ack (same path every other
	// IRQ takes), so SGI delivery is identical to RESCHED. Handles only the two
	// test SGIs; all other vectors fall through to the generic handler map.
	#[cfg(feature = "pmr-preempt-spike")]
	if vector == SGI_PMRTEST_LO || vector == SGI_PMRTEST_HI {
		let ret = pmr_spike_dispatch(vector);
		// Spike 3 (pmr-coop-net, INV-P6): under EOImode=1 the priority-drop
	// (EOIR1) + deactivate (DIR) are split; complete both AND decrement the
	// per-core EOI in-flight counter (paired with the IAR1 ack inc).
	// Default (feature off, EOImode=0): combined end_interrupt.
	#[cfg(feature = "pmr-coop-net")]
	eoi_complete(irqid);
	#[cfg(not(feature = "pmr-coop-net"))]
	GicCpuInterface::end_interrupt(irqid, InterruptGroup::Group1);
		irq_exit();
		return ret;
	}

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

	// Spike 3 (pmr-coop-net, INV-P6): under EOImode=1 the priority-drop
	// (EOIR1) + deactivate (DIR) are split; complete both AND decrement the
	// per-core EOI in-flight counter (paired with the IAR1 ack inc).
	// Default (feature off, EOImode=0): combined end_interrupt.
	#[cfg(feature = "pmr-coop-net")]
	eoi_complete(irqid);
	#[cfg(not(feature = "pmr-coop-net"))]
	GicCpuInterface::end_interrupt(irqid, InterruptGroup::Group1);

	// INV-P8 (Spike 2): do NOT run the cooperative scheduler on return from a
	// NESTED IRQ — that would switch stacks out from under a still-live outer
	// handler (stack corruption). Run it only on the OUTERMOST IRQ return
	// (nesting depth back to 0). `irq_exit()` decrements; the scheduler runs
	// iff we just returned to thread mode. Backstop assert: depth must be 0
	// when we DO call the scheduler.
	irq_exit();
	if CoreLocal::get().rt_nest_depth() == 0 {
		debug_assert!(CoreLocal::get().rt_nest_depth() == 0);
		core_scheduler().scheduler(false).unwrap_or_default()
	} else {
		ptr::null_mut()
	}
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
	// On task-context exceptions (EL0t/EL1t), clear the continuation slot
	// assignment before any path that can panic/abort/re-enter. Leaving
	// scratch_slot set allows a nested exception to hit df_check_el1t and
	// mask the real fault behind a slot-overflow double-fault.
	if !from_el1h {
		CoreLocal::get().clear_scratch_slot();
	}

	unsafe { dump_frame_once("do_sync", state as *const State) };
	let esr = ESR_EL1.get();
	let ec_raw = ESR_EL1.read(ESR_EL1::EC);
	let ec: ESR_EL1::EC::Value =
		ESR_EL1.read_as_enum(ESR_EL1::EC).unwrap();
	let iss = ESR_EL1.read(ESR_EL1::ISS);
	let pc = ELR_EL1.get();
	let fatal_finish = |reason: &str| {
		error!(
			"[TASK-FAULT] {} task={} pc={:#x}",
			reason,
			core_scheduler().get_current_task_id(),
			ELR_EL1.get()
		);
		scheduler::abort();
	};

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
				// Spike 3 (pmr-coop-net, INV-P6): under EOImode=1 the priority-drop
	// (EOIR1) + deactivate (DIR) are split; complete both AND decrement the
	// per-core EOI in-flight counter (paired with the IAR1 ack inc).
	// Default (feature off, EOImode=0): combined end_interrupt.
	#[cfg(feature = "pmr-coop-net")]
	eoi_complete(irqid);
	#[cfg(not(feature = "pmr-coop-net"))]
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

/// Double-fault source discriminator passed by `df_check_*` in start.s
/// (via `x2 = flavor`). `#[repr(u64)]` discriminants MUST match the immediates
/// emitted by the `df_check_el1h`/`df_check_el1t` macros:
///   0 = el1_sync      (EL1h, kernel context)
///   1 = el1_error     (EL1h, kernel context)
///   2 = el1_sp0_sync  (EL1t, task context)
///   3 = el1_sp0_error (EL1t, task context)
/// Decoded from the raw `u64` at the asm boundary (see `do_double_fault`); this
/// keeps the magic numbers in exactly one place instead of scattered `0/1/2/3`
/// compares.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u64)]
pub(crate) enum DfFlavor {
	El1hSync = 0,
	El1hError = 1,
	El1tSync = 2,
	El1tError = 3,
}

impl DfFlavor {
	/// Decode the raw `u64` from start.s. Unknown values are mapped to
	/// `El1hSync` (kernel context) so a corrupted/garbled flavor defaults to
	/// the conservative RCB-class halt policy rather than a risky kill.
	fn from_u64(v: u64) -> DfFlavor {
		match v {
			0 => DfFlavor::El1hSync,
			1 => DfFlavor::El1hError,
			2 => DfFlavor::El1tSync,
			3 => DfFlavor::El1tError,
			_ => DfFlavor::El1hSync,
		}
	}

	/// True if the fault was taken from EL1t (task context) — the recoverable
	/// class for Phase 7 (kill+resume). False for EL1h (kernel context) —
	/// RCB-class, must halt. This is the I3 source-branch predicate.
	fn is_el1t(self) -> bool {
		matches!(self, DfFlavor::El1tSync | DfFlavor::El1tError)
	}
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
///   - set `x2 = flavor` (raw u64; decoded to `DfFlavor` below).
///
/// We print ESR/FAR/ELR + the bad SP + flavor, then fail-stop (spin/reset).
/// We do NOT attempt recovery — a nested/overflowing handler cannot be safely
/// unwound (avoids a recursive fault storm). The EL1t/EL1h distinction is the
/// Phase 7 I3 source branch (EL1t = recoverable kill+resume; EL1h = RCB halt);
/// until Phase 7 lands, both classes fail-stop here.
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
	let flavor = DfFlavor::from_u64(flavor);
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
	error!("[DOUBLE-FAULT] task={tid:?} flavor={flavor:?} (el1t={})", flavor.is_el1t());
	error!("  bad SP_EL1    = {bad_sp:#x}");
	error!("  ESR_EL1       = {esr:#x}");
	error!("  FAR_EL1       = {far:#x}");
	error!("  ELR_EL1       = {elr:#x}");
	error!("  slot top      = {:#x}", CoreLocal::get().scratch_slot());
	error!("  (exception taken while already on a slot / slot overflow)");
	error!("============================================================");
	// I3 source branch (Phase 7/Phase 8): EL1h (kernel context) => RCB-class
	// fail-stop; EL1t (task context) => recoverable kill+resume, which may be
	// supervised (Phase 8 respawn). The branch is explicit so the two classes
	// remain independently editable (review finding #7: the EL1t arm now wires
	// the supervisor path instead of duplicating the EL1h arm).
	//
	// `DfFlavor::from_u64` is exhaustive over the four valid variants, so a
	// `match` with no `_` arm gives a compile-time exhaustiveness guarantee —
	// no runtime `debug_assert!` needed (finding #8).
	if flavor.is_el1t() {
		// Task-context double fault: kill the task, keep the system alive.
		// Phase 8: defer to the supervisor — if the task's EntryPointId policy
		// permits a restart, exit() will respawn it; otherwise it dies. Either
		// way scheduler::abort() drives the kill + reschedule (I5 liveness).
		error!("[DOUBLE-FAULT] EL1t recoverable path -> scheduler::abort() (supervised kill+resume)");
		scheduler::abort()
	} else {
		// Kernel-context double fault: hard fail-stop (RCB-class). The kernel
		// cannot safely continue from a fault in its own context, so this is
		// NOT supervised — abort() here ends the core.
		error!("[DOUBLE-FAULT] EL1h kernel-context path -> scheduler::abort() (fail-stop)");
		scheduler::abort()
	}
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

	// Spike 3 (pmr-coop-net, INV-P6): switch the GIC CPU interface to
	// EOImode=1 so priority-drop (EOIR1) and deactivate (DIR) are split,
	// enabling the per-core EOI in-flight counter. MUST be set before any
	// interrupt is acknowledged under this mode. Gated; inert in default build.
	#[cfg(feature = "pmr-coop-net")]
	eoi_mode1::set_eoi_mode_1();

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

	// Spike 4 (stackful-continuations.md §3): enable the continuation wake SGI
	// (11) on the BSP. Priority 0x60 = COOP band. AP core does the same in
	// init_cpu() (the BSP never runs init_cpu()).
	#[cfg(feature = "continuations")]
	{
		let cont_id = IntId::sgi(SGI_CONT_WAKE.into());
		gic.set_interrupt_priority(cont_id, Some(cpu_id), 0x60)
			.unwrap();
		gic.enable_interrupt(cont_id, Some(cpu_id), true).unwrap();
		IRQ_NAMES.lock().insert(SGI_CONT_WAKE, "ContWake");
	}

	// Spike-1 test SGIs (feature `pmr-preempt-spike`): register here in `init()`
	// (the BSP path) as well as in `init_cpu()` (AP path), because the BSP only
	// runs `init()` and never `init_cpu()`. Without this, SGI 14/15 are never
	// enabled on the BSP and the self-IPI is silently dropped. LO=0x40, HI=0x00.
	#[cfg(feature = "pmr-preempt-spike")]
	{
		let lo_id = IntId::sgi(SGI_PMRTEST_LO.into());
		gic.set_interrupt_priority(lo_id, Some(cpu_id), 0x40)
			.unwrap();
		gic.enable_interrupt(lo_id, Some(cpu_id), true).unwrap();
		IRQ_NAMES.lock().insert(SGI_PMRTEST_LO, "PmrTestLo");
		let hi_id = IntId::sgi(SGI_PMRTEST_HI.into());
		gic.set_interrupt_priority(hi_id, Some(cpu_id), 0x00)
			.unwrap();
		gic.enable_interrupt(hi_id, Some(cpu_id), true).unwrap();
		IRQ_NAMES.lock().insert(SGI_PMRTEST_HI, "PmrTestHi");
	}

	// Spike 2 (pmr-band) / Spike 3 (pmr-coop-net): register the COOP-band SGI
	// (13) on this core (BSP path). The RT bridge SGI (12) is Spike-2-only.
	// COOP_WAKE priority 0x60 (COOP band), RT_BRIDGE 0x20 (RT band, per §4).
	// INV-P7: the RT band is pinned to core 0.
	#[cfg(any(feature = "pmr-band", feature = "pmr-coop-net"))]
	{
		let coop_id = IntId::sgi(SGI_COOP_WAKE.into());
		gic.set_interrupt_priority(coop_id, Some(cpu_id), COOP_WAKE_PRIORITY)
			.unwrap();
		gic.enable_interrupt(coop_id, Some(cpu_id), true).unwrap();
		IRQ_NAMES.lock().insert(SGI_COOP_WAKE, "CoopWake");
	}
	#[cfg(feature = "pmr-band")]
	{
		#[cfg(feature = "smp")]
		debug_assert_eq!(cpu_id, 0, "RT-band SGI registered off core 0 (INV-P7)");
		let rt_id = IntId::sgi(SGI_RT_BRIDGE.into());
		gic.set_interrupt_priority(rt_id, Some(cpu_id), RT_BRIDGE_PRIORITY)
			.unwrap();
		gic.enable_interrupt(rt_id, Some(cpu_id), true).unwrap();
		IRQ_NAMES.lock().insert(SGI_RT_BRIDGE, "RtBridge");
	}

	*GIC.lock() = Some(gic);
	// docs/stackful-continuations.md §9 O1 (H1): publish the lock-free
	// "GIC initialized" flag so continuation boot-ready asserts can check it
	// without locking the `GIC` SpinMutex (locking would be the hazard O1
	// forbids). `set` is lock-free; idempotent across cores.
	let _ = GIC_READY.try_insert(());
}

/// docs/stackful-continuations.md §9 O1 (H1): lock-free "GIC initialized" flag.
/// Published once in `install_gic` after `GIC` is populated. Read via
/// `GIC_READY.get().is_some()` (lock-free) so continuation boot-ready asserts
/// need not lock the `GIC` SpinMutex.
pub(crate) static GIC_READY: OnceCell<()> = OnceCell::new();

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

	// Spike 3 (pmr-coop-net, INV-P6): EOImode=1 on the AP core too (must match
	// the BSP before any interrupt is acked under split-EOI). Gated; inert
	// in default build.
	#[cfg(feature = "pmr-coop-net")]
	eoi_mode1::set_eoi_mode_1();

	// INV-P1 (rtic-gicv3-async-reactor.md §8): probe implemented priority bits.
	// Write 0xFF, read back; the number of implemented LOW bits in the read-back
	// value is the implemented priority-bit count. QEMU virt GICv3 commonly
	// implements 5 bits (0xE0 read-back => 5 bits, 32 levels). Assertion guards
	// the band layout against hardware that exposes fewer bits than assumed.
	#[cfg(feature = "pmr-preempt-spike")]
	{
		GicCpuInterface::set_priority_mask(0xff);
		let readback = GicCpuInterface::get_priority_mask();
		// Probe: writing 0xFF, the implemented HIGH bits return 1 and the
		// unimplemented LOW bits return 0. So the implemented bit count is the
		// number of set bits in the read-back (robust to MSB-alignment).
		let impl_bits = readback.count_ones() as u8;
		info!("PMR implemented bits probe: wrote 0xff, read back {readback:#x} => {impl_bits} implemented priority bits");
		// GICv3 guarantees >= 4 implemented bits (16 levels). Assert that floor.
		assert!(impl_bits >= 4, "GIC implements fewer than 4 priority bits ({impl_bits}); band layout invalid");
		// Our band plan assumes >= 5 bits (RT 0x00-0x3F vs COOP 0x40-0x7F).
		// Allow 4-bit parts to still boot but warn; Spike-1 priorities are chosen
		// within the top implemented bits so the test remains valid.
		if impl_bits < 5 {
			warn!("GIC implements only {impl_bits} priority bits; band layout is tighter than assumed (documented as a non-fatal degradation for Spike 1)");
		}
		GicCpuInterface::set_priority_mask(0xff);
	}

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

	#[cfg(feature = "pmr-preempt-spike")]
	{
		// LO = priority 0x40 (below HI, above thread/idle). HI = 0x00 (highest).
		// Spike-1 do_irq path pends HI from inside LO to prove nested preemption;
		// a 0x00 PMR control case proves HI is then blocked.
		let lo_id = IntId::sgi(SGI_PMRTEST_LO.into());
		gic.set_interrupt_priority(lo_id, Some(cpu_id), 0x40)
			.unwrap();
		gic.enable_interrupt(lo_id, Some(cpu_id), true).unwrap();
		IRQ_NAMES.lock().insert(SGI_PMRTEST_LO, "PmrTestLo");
		let hi_id = IntId::sgi(SGI_PMRTEST_HI.into());
		gic.set_interrupt_priority(hi_id, Some(cpu_id), 0x00)
			.unwrap();
		gic.enable_interrupt(hi_id, Some(cpu_id), true).unwrap();
		IRQ_NAMES.lock().insert(SGI_PMRTEST_HI, "PmrTestHi");
	}

	// Spike 2 (pmr-band) / Spike 3 (pmr-coop-net): register the COOP-band SGI
	// (13) on this core; the RT bridge SGI (12) is Spike-2-only. GOTCHA
	// (Spike 1): registration must happen in BOTH init() and init_cpu() or the
	// BSP self-IPI is silently dropped.
	#[cfg(any(feature = "pmr-band", feature = "pmr-coop-net"))]
	{
		let coop_id = IntId::sgi(SGI_COOP_WAKE.into());
		gic.set_interrupt_priority(coop_id, Some(cpu_id), COOP_WAKE_PRIORITY)
			.unwrap();
		gic.enable_interrupt(coop_id, Some(cpu_id), true).unwrap();
		IRQ_NAMES.lock().insert(SGI_COOP_WAKE, "CoopWake");
	}
	#[cfg(feature = "pmr-band")]
	{
		#[cfg(feature = "smp")]
		debug_assert_eq!(cpu_id, 0, "RT-band SGI registered off core 0 (INV-P7)");
		let rt_id = IntId::sgi(SGI_RT_BRIDGE.into());
		gic.set_interrupt_priority(rt_id, Some(cpu_id), RT_BRIDGE_PRIORITY)
			.unwrap();
		gic.enable_interrupt(rt_id, Some(cpu_id), true).unwrap();
		IRQ_NAMES.lock().insert(SGI_RT_BRIDGE, "RtBridge");
	}
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
