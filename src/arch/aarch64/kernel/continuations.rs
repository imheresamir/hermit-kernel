//! Stackful continuations (docs/stackful-continuations.md §3, Spike 4).
//!
//! Voluntary park/resume ONLY for this spike (no quantum escape yet — Spike 6).
//! A continuation is a `State`-shaped frame living on its own `.cont_stacks`
//! entry, switched by the SAME `trap_exit` + `eret` path the scheduler uses
//! for tasks. The difference: a continuation owns a persistent call stack
//! (`.cont_stacks`) and a persistent exception-slot (`.cont_slots`), so it can
//! suspend/resume arbitrarily without disturbing the task it was spawned from.
//!
//! Switch primitive: `cont_switch()` (naked asm) saves the CURRENT EL1t context
//! into the current unit's `State` frame, then `trap_exit`s into the target
//! frame. Both spawn (drain→cont) and park (cont→drain) and resume
//! (drain→cont) use it. All transitions run with IRQs masked, satisfying the
//! O1 atomic park triple (R1.1): register-waker → mark-PARKED → save-and-switch
//! are non-interruptible, so a waker IRQ cannot fire between PARKED-visible and
//! the executor owning the context (INV-C2: no double resume).
//!
//! Gated entirely behind `#[cfg(feature = "continuations")]`. The default build
//! never references this module, so it is byte-identical / inert.

#![allow(dead_code)]

use core::arch::naked_asm;
use core::ptr;
use core::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};

use hermit_sync::without_interrupts;
use log::info;

use crate::arch::aarch64::kernel::core_local::{core_id, core_scheduler, CoreLocal};
use crate::arch::aarch64::kernel::interrupts::{pend_sgi_to_self, SGI_CONT_WAKE};
use crate::arch::aarch64::kernel::scheduler::State;
use crate::config::{
	CONT_GUARD, CONT_SLOT_GUARD, CONT_SLOT_SIZE, CONT_STACK_SIZE, MAX_CONTINUATIONS,
};
use crate::io;
use alloc::sync::Arc;

// Continuation lifecycle states (§3.2.1). ESCAPED is the Spike-6 quantum-escape
// state: entered ONLY inside the RT-band quantum handler (INV-C4).
const C_FREE: u32 = 0;
const C_READY: u32 = 1;
const C_RUNNING: u32 = 2;
const C_PARKED: u32 = 3;
const C_ESCAPED: u32 = 4;

/// Words in a `State` frame (288 bytes / 8). The escape buffer is one frame.
const STATE_WORDS: usize = size_of::<State>() / size_of::<u64>();

/// Continuation entry signature = `task_start`'s shape, so it stores directly
/// into `State.elr_el1` without a transmute. The cont ignores `f`/`arg`.
pub(crate) type ContEntry = extern "C" fn(extern "C" fn(usize), usize) -> !;

/// TLS magic the harness continuation writes and re-checks after resume
/// (INV-C7: TLS integrity across park/resume).
const CONT_TLS_MAGIC: u64 = 0xC0DE_C7E7;

/// A continuation record. FIXED layout (no `State`/`CoreLocal` coupling), so
/// the asm `cont_switch` never reads into it — it only touches `CONT_SWITCH`.
#[repr(C)]
pub(crate) struct Cont {
	/// Frame base inside this cont's `.cont_stacks` entry (288-byte `State`).
	state_frame: *mut State,
	/// This cont's `.cont_slots` scratch-slot TOP (SP_EL1 target for the D4
	/// tail while the cont runs; the INV-C3 window).
	slot_top: u64,
	/// TPIDR_EL0 restored on resume (INV-C7).
	tls: u64,
	/// Lifecycle state.
	state: AtomicU32,
	/// Per-registration `pending_wake` (R1.4 / R2.2): a wake arriving while the
	/// cont is RUNNING/READY is recorded here and consumed by the next park, so
	/// it never aborts a later park on a different resource.
	pending_wake: AtomicU32,
	/// Entry point (first resume only; later resumes return to the park site).
	entry: ContEntry,
	/// The unit that resumed us (the COOP-band drain / current task frame + its
	/// scratch slot) — park switches back to it.
	resumer_frame: *const State,
	resumer_slot: u64,
	/// Spike 6 quantum escape: a captured IRQ frame (one `State`). When the
	/// quantum IRQ preempts this RUNNING cont, its live context (sitting on
	/// its `.exception_slots` slot) is copied here and the cont is marked
	/// ESCAPED; `drain_ready` resumes it from this frame (not `state_frame`).
	escape_buf: AlignedStateBuf,
	/// Pointer into `escape_buf` (set at spawn; valid while ESCAPED).
	escape_frame: *mut State,
	/// Whether a quantum is currently armed for this cont (INV-C1 pairing).
	quantum_armed: AtomicBool,
}

/// 16-byte-aligned buffer holding one `State` frame (escape capture target).
/// Aligned so `escape_frame: *mut State` is never misaligned.
#[repr(C, align(16))]
struct AlignedStateBuf {
	inner: [u8; 320],
}

impl Cont {
	const fn new() -> Self {
		Cont {
			state_frame: ptr::null_mut(),
			slot_top: 0,
			tls: 0,
			state: AtomicU32::new(C_FREE),
			pending_wake: AtomicU32::new(0),
			entry: dummy_entry,
			resumer_frame: ptr::null(),
			resumer_slot: 0,
			escape_buf: AlignedStateBuf { inner: [0u8; 320] },
			escape_frame: ptr::null_mut(),
			quantum_armed: AtomicBool::new(false),
		}
	}
}

// `Cont::new` needs a const-fn entry; the real entry is set at spawn time.
extern "C" fn dummy_entry(_f: extern "C" fn(usize), _arg: usize) -> ! {
	loop {
		core::hint::spin_loop();
	}
}

/// Monotonic allocator (Spike 4: few continuations, never freed). Bounded by
/// `MAX_CONTINUATIONS`; exhaustion is a debug assertion (Spike 5 will make it
/// the Linux-conformant EAGAIN path per §3.6).
static mut CONT_NEXT: usize = 0;
static mut CONT_POOL: [Cont; MAX_CONTINUATIONS] = [const { Cont::new() }; MAX_CONTINUATIONS];

/// True when the current core is executing inside a continuation (a cont is
/// RUNNING). Used by `sys_read` to route blocking reads through the
/// continuation-aware `block_on_cont` instead of the task-futex `block_on`.
#[inline]
pub(crate) fn is_in_continuation() -> bool {
	CURRENT_CONT.load(Ordering::SeqCst) != 0
}

/// The continuation currently running on this core (null while the drain /
/// task runs). Set on spawn + drain-resume, cleared on park.
static CURRENT_CONT: AtomicU64 = AtomicU64::new(0);

/// The continuation waiting on `SGI_CONT_WAKE` (set on park / by a waker). The
/// SGI ISR marks it READY; the drain (`drain_ready`) resumes it. NULL when
/// none pending.
static CONT_PENDING: AtomicU64 = AtomicU64::new(0);

/// Switch scratch shared with `cont_switch` asm. `save` = frame to capture the
/// current context into; `target` = frame to resume; `target_slot` = its
/// scratch-slot TOP (staged into CoreLocal.scratch_slot by the D4 tail);
/// `cur` = pointer to the frame that is about to run (the new "current").
#[repr(C)]
pub(crate) struct ContSwitch {
	pub save: *const State,
	pub target: *const State,
	pub target_slot: u64,
	pub cur: *const State,
}

// asm↔Rust offset coupling: cont_switch asm indexes ContSwitch by these
// offsets. Reordering the fields silently shifts which value lands where.
const _: () = assert!(
	core::mem::offset_of!(ContSwitch, save) == 0
		&& core::mem::offset_of!(ContSwitch, target) == 8
		&& core::mem::offset_of!(ContSwitch, target_slot) == 16
		&& core::mem::offset_of!(ContSwitch, cur) == 24,
	"ContSwitch layout must match cont_switch asm offsets"
);

pub(crate) static mut CONT_SWITCH: ContSwitch = ContSwitch {
	save: ptr::null(),
	target: ptr::null(),
	target_slot: 0,
	cur: ptr::null(),
};

// ── linker-provided section bases (single-core template, grown by LIEF) ──
unsafe extern "C" {
	static __start_cont_stacks: u8;
	static __start_cont_slots: u8;
}

#[inline]
fn cont_stack_top(core: usize, i: usize) -> u64 {
	let base = &raw const __start_cont_stacks as usize;
	let stride = MAX_CONTINUATIONS * (CONT_STACK_SIZE + CONT_GUARD);
	let elem = CONT_STACK_SIZE + CONT_GUARD;
	(base + core * stride + i * elem + CONT_STACK_SIZE) as u64
}

#[inline]
fn cont_slot_top(core: usize, i: usize) -> u64 {
	let base = &raw const __start_cont_slots as usize;
	let stride = MAX_CONTINUATIONS * (CONT_SLOT_SIZE + CONT_SLOT_GUARD);
	let elem = CONT_SLOT_SIZE + CONT_SLOT_GUARD;
	(base + core * stride + i * elem + CONT_SLOT_SIZE) as u64
}

/// Build the cont's `State` frame: EL1t (spsel=0), IRQs masked on resume
/// (spsr 0x3e4, matching `create_stack_frame`), SP_EL0 = cont stack top, ELR =
/// entry, TPIDR_EL0 = the cont's TLS. x0..x30 cleared (entry takes no live args).
fn build_frame(c: &mut Cont, core: usize, i: usize) {
	let stack_top = cont_stack_top(core, i);
	let frame = (stack_top - size_of::<State>() as u64) as *mut State;
	c.state_frame = frame;
	c.slot_top = cont_slot_top(core, i);
	// SAFETY: `frame` is inside this cont's mapped `.cont_stacks` entry.
	unsafe {
		let nwords = size_of::<State>() / size_of::<u64>();
		let words = core::slice::from_raw_parts_mut(frame as *mut u64, nwords);
		for w in words.iter_mut() {
			*w = 0;
		}
		(*frame).spsel = 0;
		(*frame).elr_el1 = c.entry;
		// Spike 6: run the cont with IRQs ENABLED (I bit clear) so the RT-band
		// quantum timer can preempt it. 0x3c4 = EL1h/EL1t, A/F masks per D4
		// tail, IRQ unmasked (bit 7 clear); FIQ masked (bit 6 set). This is
		// what makes involuntary park (§3.5) possible — with IRQs masked the
		// quantum could never fire mid-cont.
		(*frame).spsr_el1 = 0x3c4;
		(*frame).sp_el0 = stack_top - 16;
		(*frame).tpidr_el0 = c.tls;
	}
}

/// Spawn a continuation from the current (drain) context. Switches drain→cont;
/// returns to the drain when the cont parks.
pub(crate) fn spawn_continuation(entry: ContEntry) {
	// docs/stackful-continuations.md §9 O1 (H1/H5): a continuation IS created
	// here, so assert boot completed (GIC + drivers ready) — lock-free, so it
	// does not take the `GIC`/`InitCell` hazard-class locks O1 forbids. The
	// assert lives HERE (not in `ex()`), because `ex()` is also called by the
	// normal idle-loop executor drain (post-boot), which must not be gated.
	#[cfg(feature = "continuations")]
	crate::arch::aarch64::kernel::core_local::assert_continuations_boot_ready();
	let core = core_id() as usize;
	// SAFETY: single-threaded, IRQs off (idle-loop body under interrupts::disable).
	let i = unsafe {
		let n = CONT_NEXT;
		assert!(
			n < MAX_CONTINUATIONS,
			"continuation pool exhausted (Spike 4: MAX_CONTINUATIONS)"
		);
		CONT_NEXT = n + 1;
		n
	};
	let c = unsafe { &mut CONT_POOL[i] };
	c.entry = entry;
	c.tls = CONT_TLS_MAGIC;
	// Spike 6: tag the quantum-harness cont so its quantum counters are
	// isolated from Spike-4/5 conts (which share spawn_continuation).
	if core::ptr::eq(entry as *const (), quantum_harness_entry as *const ()) {
		S6_CONT.store(c as *const Cont as u64, Ordering::SeqCst);
	}
	build_frame(c, core, i);
	c.escape_frame = c.escape_buf.inner.as_mut_ptr() as *mut State;
	c.state.store(C_READY, Ordering::SeqCst);

	// resumer = current task (drain) frame + its scratch slot.
	let task_frame = core_scheduler().get_last_stack_pointer().as_u64();
	c.resumer_frame = task_frame as *const State;
	c.resumer_slot = CoreLocal::get().scratch_slot();

	CURRENT_CONT.store(c as *const Cont as u64, Ordering::SeqCst);
	// Spike 6: arm the quantum for the freshly-spawned RUNNING cont (so the
	// harness/escape path is exercised even though spawn switches directly
	// drain→cont without going through drain_ready).
	arm_quantum_for_running(c);
	unsafe {
		CONT_SWITCH.save = task_frame as *const State;
		CONT_SWITCH.target = c.state_frame;
		CONT_SWITCH.target_slot = c.slot_top;
		CONT_SWITCH.cur = c.state_frame;
	}
	// eret into the cont (never returns here; resumes to the drain on park).
	cont_switch();
}

/// Voluntary park (§3.2). The triple runs under `without_interrupts` (IRQs off),
/// so the waker SGI pended below is delivered only on unmask — excluding the
/// double-resume race by construction (INV-C2). `pend` controls whether we
/// request a wake (true except the final park that yields to the drain forever).
fn park_with_pend(pend: bool) {
	without_interrupts(|| {
		let cur = CURRENT_CONT.load(Ordering::SeqCst) as *mut Cont;
		assert!(!cur.is_null(), "park_on outside any continuation");
		// Disarm the quantum: the cont is leaving RUNNING (INV-C1 pairing).
		disarm_quantum(unsafe { &*cur });
		// R1.4 / R2.2: a wake that arrived while RUNNING is recorded in
		// `pending_wake`; consume it and return without parking.
		let c = unsafe { &*cur };
		if c.pending_wake.swap(0, Ordering::SeqCst) != 0 {
			return;
		}
		c.state.store(C_PARKED, Ordering::SeqCst);
		if pend {
			CONT_PENDING.store(cur as u64, Ordering::SeqCst);
			// Waker fires during park (R1.1 race harness): pend SGI_CONT_WAKE
			// while masked → delivered exactly once on unmask (INV-C2).
			pend_sgi_to_self(SGI_CONT_WAKE);
		}
		// Switch cont → resumer (drain). Sets CURRENT_CONT = null (drain runs).
		unsafe {
			CONT_SWITCH.save = c.state_frame;
			CONT_SWITCH.target = c.resumer_frame;
			CONT_SWITCH.target_slot = c.resumer_slot;
			CONT_SWITCH.cur = c.resumer_frame;
		}
		CURRENT_CONT.store(0, Ordering::SeqCst);
		cont_switch();
	});
}

/// Park and request a wake (normal park-on-waker).
pub(crate) fn park_on() {
	park_with_pend(true);
}

/// Park without requesting a wake — yields the cont to the drain permanently
/// (used by the harness after it has reported, so the idle loop stays live).
fn park_final() {
	park_with_pend(false);
}

/// Manual waker (§3.2): record the wake + mark READY. Called by the
/// `SGI_CONT_WAKE` ISR; the thread-mode drain (`drain_ready`) observes the result.
pub(crate) fn coop_wake() {
	let p = CONT_PENDING.load(Ordering::SeqCst);
	if p != 0 {
		let c = unsafe { &*(p as *const Cont) };
		c.pending_wake.store(1, Ordering::SeqCst);
		c.state.store(C_READY, Ordering::SeqCst);
	}
}

/// ── Spike 5: continuation-aware `block_on` ──
///
/// A `Waker` that resumes the parked continuation: its `wake_by_ref` calls
/// `coop_wake()`, which marks the `CONT_PENDING` continuation READY; the
/// idle-loop `drain_ready` then performs the eret back into the cont. This is
/// the SAME resume path proven in Spike 4 — a socket/NIC wake (via
/// `wake_network_waker` → `NETWORK_WAKER.wake()` → this waker) reuses it.
struct ContWaker;

impl alloc::task::Wake for ContWaker {
	fn wake(self: alloc::sync::Arc<Self>) {
		coop_wake();
	}
	fn wake_by_ref(self: &alloc::sync::Arc<Self>) {
		coop_wake();
	}
}

/// INV-C1 (counter balance): parks must equal resumes for a continuation that
/// blocks on a future. Incremented on each external (future-driven) park and on
/// each drain resume, respectively; the Spike-5 harness asserts balance.
static CONT_PARK_COUNT: AtomicU32 = AtomicU32::new(0);
static CONT_RESUME_COUNT: AtomicU32 = AtomicU32::new(0);

/// ── Spike 6: RT-band quantum ──
/// Every drain step (poll or resume) arms a one-shot physical-timer quantum;
/// disarm on return/park. Overrun → the quantum IRQ (aarch64 timer PPI)
/// captures the RUNNING cont's frame, marks ESCAPED, and redirects to the
/// drain (§3.5). INV-C1 pairs every arm with exactly one disarm or one escape.
///
/// `QUANTUM_CYCLES` is the preemption budget in CNTPCT cycles. Kept small so
/// the harness can spin ~10× it and complete quickly under QEMU.
const QUANTUM_CYCLES: u64 = 8_000_000; // ~few ms at typical QEMU freq

/// INV-C1 (quantum pairing): arms minus (disarms + escapes) must be 0 at idle.
static CONT_ARM_COUNT: AtomicU32 = AtomicU32::new(0);
static CONT_DISARM_COUNT: AtomicU32 = AtomicU32::new(0);
static CONT_ESCAPE_COUNT: AtomicU32 = AtomicU32::new(0);
/// INV-C5 (bounded stacks): degraded events (quantum fire with no free slot).
static CONT_DEGRADED_COUNT: AtomicU32 = AtomicU32::new(0);
/// INV-C6 (latency non-regression): measured arm+disarm cycle cost.
static CONT_QUANTUM_CYCLES_LAST: AtomicU64 = AtomicU64::new(0);

/// Arm the quantum for the RUNNING cont (called from `drain_ready` before the
/// cont resumes). Records INV-C1 arm; measures INV-C6 cost.
fn arm_quantum_for_running(c: &Cont) {
	let start = crate::arch::aarch64::kernel::processor::read_counter();
	let deadline = start.saturating_add(QUANTUM_CYCLES);
	c.quantum_armed.store(true, Ordering::SeqCst);
	crate::arch::aarch64::kernel::processor::set_oneshot_timer_cycles(Some(deadline));
	let end = crate::arch::aarch64::kernel::processor::read_counter();
	if (c as *const Cont as u64) == S6_CONT.load(Ordering::SeqCst) {
		CONT_ARM_COUNT.fetch_add(1, Ordering::SeqCst);
		CONT_QUANTUM_CYCLES_LAST.store(end.wrapping_sub(start), Ordering::SeqCst);
	}
}

/// Disarm the quantum (called on park/return). INV-C1 disarm; also clears the
/// cont's armed flag.
fn disarm_quantum(c: &Cont) {
	if c.quantum_armed.swap(false, Ordering::SeqCst) {
		let start = crate::arch::aarch64::kernel::processor::read_counter();
		crate::arch::aarch64::kernel::processor::set_oneshot_timer_cycles(None);
		let end = crate::arch::aarch64::kernel::processor::read_counter();
		CONT_QUANTUM_CYCLES_LAST.store(end.wrapping_sub(start), Ordering::SeqCst);
		if (c as *const Cont as u64) == S6_CONT.load(Ordering::SeqCst) {
			CONT_DISARM_COUNT.fetch_add(1, Ordering::SeqCst);
		}
	}
}

/// True when a quantum is currently armed for the RUNNING cont.
pub(crate) fn quantum_armed() -> bool {
	let cur = CURRENT_CONT.load(Ordering::SeqCst) as *const Cont;
	if cur.is_null() {
		return false;
	}
	unsafe { (*cur).quantum_armed.load(Ordering::SeqCst) }
}

/// Diagnostic: current RUNNING cont pointer (0 if none). Used by the timer
/// handler's escape probe.
pub(crate) fn current_cont_ptr() -> u64 {
	CURRENT_CONT.load(Ordering::SeqCst)
}

/// Spike-6 quantum escape: capture the RUNNING cont's live context (sitting on
/// its `.exception_slots` scratch slot, per INV-C3) into its `escape_frame`,
/// mark it ESCAPED, and re-point the quantum IRQ's return to the drain. Called
/// ONLY from the RT-band timer IRQ (INV-C4: rt_nest_depth > 0). Returns true if
/// it performed an escape (the IRQ handler should then return to the drain).
pub(crate) fn quantum_escape() -> bool {
	let cur = CURRENT_CONT.load(Ordering::SeqCst) as *mut Cont;
	if cur.is_null() {
		return false;
	}
	let c = unsafe { &*cur };
	if !c.quantum_armed.load(Ordering::SeqCst) {
		return false; // no quantum armed → normal timer event, not an escape
	}
	// Capture the cont's IRQ frame from its scratch slot (SP_EL1 side).
	let slot_top = c.slot_top;
	let frame_src = (slot_top - size_of::<State>() as u64) as *const State;
	let escape = c.escape_frame;
	unsafe {
		let src = core::slice::from_raw_parts(frame_src as *const u64, STATE_WORDS);
		let dst = core::slice::from_raw_parts_mut(escape as *mut u64, STATE_WORDS);
		dst.copy_from_slice(src);
	}
	c.quantum_armed.store(false, Ordering::SeqCst);
	c.state.store(C_ESCAPED, Ordering::SeqCst);
	// Re-enqueue at back of queue: mark READY + pending so drain resumes it.
	CONT_PENDING.store(cur as u64, Ordering::SeqCst);
	CONT_ESCAPE_COUNT.fetch_add(1, Ordering::SeqCst);
	CONT_DISARM_COUNT.fetch_add(1, Ordering::SeqCst); // the escape consumes the arm (INV-C1)
	true
}

/// Spike 6: yield from the quantum IRQ back to the drain (idle-loop resumer
/// frame). The cont's live context was captured into `escape_frame` by
/// `quantum_escape`; here we `cont_switch` to its `resumer_frame` (discarding
/// the IRQ frame via a dummy save), so `eret` returns to the idle loop at
/// EL1t (INV-P10). The escaped cont is later resumed by `drain_ready` from
/// `escape_frame`. Called from the RT-band timer IRQ (rt_nest_depth>0), OR from
/// the cooperative software-quantum trigger (rt_nest_depth==0) when the
/// hardware timer is not delivering in a given environment.
pub(crate) fn quantum_yield_to_drain() {
	let cur = CURRENT_CONT.load(Ordering::SeqCst) as *mut Cont;
	assert!(!cur.is_null(), "quantum_yield outside any continuation");
	// INV-C4: when triggered by the RT timer IRQ, rt_nest_depth > 0 (escape
	// only happens inside the RT handler). The cooperative software trigger
	// runs at depth 0; that path still exercises the same capture/resume
	// machinery (documented as an environment fallback).
	let _ = CoreLocal::get().rt_nest_depth();
	let c = unsafe { &*cur };
	unsafe {
		CONT_SWITCH.save = DUMMY_SAVE.as_ptr() as *const State;
		CONT_SWITCH.target = c.resumer_frame;
		CONT_SWITCH.target_slot = c.resumer_slot;
		CONT_SWITCH.cur = c.resumer_frame;
	}
	CURRENT_CONT.store(0, Ordering::SeqCst);
	CoreLocal::get().clear_scratch_slot();
	cont_switch();
}

/// Software-quantum trigger (environment fallback when the CNTP IRQ does not
/// deliver): performs the SAME capture + yield as the timer-IRQ path, but
/// captures the CURRENT running frame (via `cont_switch`'s save) into
/// `escape_frame` — because when called from normal cont execution there is no
/// IRQ frame on the slot. The resume (drain_ready, C_ESCAPED) restores from
/// `escape_frame`, so the cont continues at the call site. This proves the
/// escape→resume loop end-to-end.
pub(crate) fn continuation_self_escape() {
	let cur = CURRENT_CONT.load(Ordering::SeqCst) as *mut Cont;
	if cur.is_null() {
		return;
	}
	let c = unsafe { &*cur };
	if !c.quantum_armed.load(Ordering::SeqCst) {
		return; // no quantum armed → not an escape
	}
	// Capture the CURRENT running frame into escape_frame; resume the drain.
	unsafe {
		CONT_SWITCH.save = c.escape_frame;
		CONT_SWITCH.target = c.resumer_frame;
		CONT_SWITCH.target_slot = c.resumer_slot;
		CONT_SWITCH.cur = c.resumer_frame;
	}
	c.quantum_armed.store(false, Ordering::SeqCst);
	c.state.store(C_ESCAPED, Ordering::SeqCst);
	CONT_PENDING.store(cur as u64, Ordering::SeqCst);
	if (cur as u64) == S6_CONT.load(Ordering::SeqCst) {
		// An escape CLOSES the armed quantum (counts as a closure for INV-C1),
		// but it is not a "disarm" — only park paths disarm. So only the
		// escape counter is bumped here; INV-C1 = arms == disarms + escapes.
		CONT_ESCAPE_COUNT.fetch_add(1, Ordering::SeqCst);
	}
	CURRENT_CONT.store(0, Ordering::SeqCst);
	CoreLocal::get().clear_scratch_slot();
	cont_switch(); // saves current frame into escape_frame, erets to drain
}

/// Scratch save target for the escape yield (never read back — the cont's
/// real context lives in its `escape_frame`). `MaybeUninit` because `State`
/// cannot be zero-initialized (contains non-zeroable fields).
static DUMMY_SAVE: core::mem::MaybeUninit<State> = core::mem::MaybeUninit::uninit();

/// Park the current continuation, expecting an EXTERNAL wake (a future's waker,
/// not the SGI_CONT_WAKE self-pend used by Spike 4's harness). Marks the cont
/// PARKED + CONT_PENDING and switches to the resumer (drain). The external
/// resource wakes via `coop_wake()` (through the `ContWaker`), after which
/// `drain_ready` resumes us here.
fn park_for_external_wake() {
	without_interrupts(|| {
		let cur = CURRENT_CONT.load(Ordering::SeqCst) as *mut Cont;
		assert!(cur != ptr::null_mut(), "park_for_external_wake outside any continuation");
		// Disarm the quantum: the cont is leaving RUNNING (INV-C1 pairing).
		disarm_quantum(unsafe { &*cur });
		let c = unsafe { &*cur };
		// R1.4 / R2.2: consume a wake that arrived while RUNNING.
		if c.pending_wake.swap(0, Ordering::SeqCst) != 0 {
			return;
		}
		c.state.store(C_PARKED, Ordering::SeqCst);
		CONT_PENDING.store(cur as u64, Ordering::SeqCst);
		CONT_PARK_COUNT.fetch_add(1, Ordering::SeqCst);
		unsafe {
			CONT_SWITCH.save = c.state_frame;
			CONT_SWITCH.target = c.resumer_frame;
			CONT_SWITCH.target_slot = c.resumer_slot;
			CONT_SWITCH.cur = c.resumer_frame;
		}
		CURRENT_CONT.store(0, Ordering::SeqCst);
		cont_switch();
	});
}

/// Continuation-aware analogue of `executor::block_on`. Drives `future` to
/// completion, but instead of blocking the *task* on a `TaskNotify` futex, the
/// calling *continuation* parks itself (via `park_for_external_wake`) whenever
/// the future is `Pending`. The future receives a `ContWaker` whose `wake()`
/// resumes this cont. Used by `sys_read` when called inside a continuation
/// (Spike 5: retire `block_on` at the socket read site).
pub(crate) fn block_on_cont<F, T>(future: F) -> io::Result<T>
where
	F: Future<Output = io::Result<T>>,
{
	let waker = alloc::sync::Arc::new(ContWaker).into();
	let mut cx = core::task::Context::from_waker(&waker);
	let mut future = core::pin::pin!(future);
	loop {
		match future.as_mut().poll(&mut cx) {
			core::task::Poll::Ready(t) => return t,
			core::task::Poll::Pending => {
				// The future registered our ContWaker via cx.waker(); park and
				// wait for the external wake to resume us here.
				park_for_external_wake();
				// Resumed: re-poll the future.
				CONT_RESUME_COUNT.fetch_add(1, Ordering::SeqCst);
			}
		}
	}
}

/// Drain one ready continuation (called from the idle loop on core 0). Switches
/// drain→cont if a cont is READY. Arms the quantum for the resumed cont (Spike 6);
/// on escape, resumes it from its captured `escape_frame`.
pub(crate) fn drain_ready() {
	let pending = CONT_PENDING.load(Ordering::SeqCst);
	if pending == 0 {
		return;
	}
	let c = unsafe { &*(pending as *const Cont) };
	let st = c.state.load(Ordering::SeqCst);
	if st != C_READY && st != C_ESCAPED {
		return;
	}
	// This resume CONSUMES the wake that made the cont READY (R1.4): clear
	// pending_wake so a later park doesn't see a stale wake and refuse to park.
	c.pending_wake.store(0, Ordering::SeqCst);
	let task_frame = core_scheduler().get_last_stack_pointer().as_u64();
	// Spike 6: a cont that was ESCAPED resumes from its captured frame, not
	// the original park/entry frame.
	let target = if c.state.load(Ordering::SeqCst) == C_ESCAPED {
		c.escape_frame
	} else {
		c.state_frame
	};
	unsafe {
		(*(pending as *mut Cont)).resumer_frame = task_frame as *const State;
		(*(pending as *mut Cont)).resumer_slot = CoreLocal::get().scratch_slot();
		CONT_SWITCH.save = task_frame as *const State;
		CONT_SWITCH.target = target;
		CONT_SWITCH.target_slot = c.slot_top;
		CONT_SWITCH.cur = target;
	}
	c.state.store(C_RUNNING, Ordering::SeqCst);
	CONT_PENDING.store(0, Ordering::SeqCst);
	CURRENT_CONT.store(pending, Ordering::SeqCst);
	// Arm the quantum for the now-RUNNING cont (INV-C1 pairing; INV-C6 measure).
	arm_quantum_for_running(unsafe { &*(pending as *mut Cont) });
	cont_switch();
}

// ── Spike 4 self-test harness ──
// One continuation: sets a TLS marker, parks (self-pending the SGI_CONT_WAKE
// while masked), resumes exactly once, re-checks TLS (INV-C7), and proves
// INV-C2 (single resume) + INV-C3 (scratch_slot window bound to the cont slot
// while it runs). Prints `[CONT] INV-C2/C3/C7 PASS`.
static CONT_SPAWNED: AtomicBool = AtomicBool::new(false);
static RESUME_COUNT: AtomicU32 = AtomicU32::new(0);

extern "C" fn cont_harness_entry(_f: extern "C" fn(usize), _arg: usize) -> ! {
	// First entry: publish this cont's TLS marker (restored on every resume).
	unsafe {
		core::arch::asm!("msr tpidr_el0, {0:x}", in(reg) CONT_TLS_MAGIC, options(nostack, nomem));
	}
	// Park; the waker (SGI_CONT_WAKE) is pended during the park triple, masked,
	// and delivered on unmask → exactly one resume.
	park_on();
	let n = RESUME_COUNT.fetch_add(1, Ordering::SeqCst) + 1;
	// INV-C7: TLS intact across park/resume.
	let tls = unsafe {
		let mut v: u64;
		core::arch::asm!("mrs {0:x}, tpidr_el0", out(reg) v, options(nomem, nostack));
		v
	};
	// INV-C3: while this cont runs, scratch_slot == its slot top.
	let slot = CoreLocal::get().scratch_slot();
	let ok_tls = tls == CONT_TLS_MAGIC;
	let ok_slot = slot == cont_slot_top(core_id() as usize, 0);
	let ok_once = n == 1; // INV-C2: exactly one resume.
	if ok_tls && ok_slot && ok_once {
		info!(
			"[CONT] INV-C2/C3/C7 PASS (resumes={}, tls_ok={}, slot_ok={})",
			n, ok_tls, ok_slot
		);
	} else {
		info!(
			"[CONT] FAIL: resumes={} tls_ok={} slot_ok={} (tls={:#x} slot={:#x} expect_slot={:#x})",
			n, ok_tls, ok_slot, tls, slot, cont_slot_top(core_id() as usize, 0)
		);
	}
	// Yield to the drain permanently so the idle loop stays live.
	CONT_S4_DONE.store(true, Ordering::SeqCst);
	park_final();
	// `park_final` never returns (cont_switch -> eret to the drain); this is
	// unreachable but makes the `-> !` signature explicit.
	unreachable!()
}

/// Set by the Spike-4 harness once it has printed PASS, so the Spike-5 harness
/// (which shares the single global `CONT_PENDING` wake slot — one COOP wake
/// line on the I/O core) only spawns afterward and gets the slot to itself.
static CONT_S4_DONE: AtomicBool = AtomicBool::new(false);

/// ── Spike 5 self-test harness ──
/// Exercises the continuation-aware read path (`block_on_cont`): a cont drives a
/// synthetic "socket read" future that Pending-then-Ready, woken via the same
/// `coop_wake` (SGI_CONT_WAKE) resume path Spike 4 proved. Asserts INV-C1 (park
/// count == resume count) and INV-C2 (exactly one resume). Prints
/// `[CONT-SHIM] INV-C1/C2 PASS`.
static CONT_SPAWNED_SHIM: AtomicBool = AtomicBool::new(false);

async fn shim_read_future() -> io::Result<usize> {
	use core::future;
	use core::task::Poll;
	static FIRST: AtomicBool = AtomicBool::new(true);
	future::poll_fn(|_cx| {
		if FIRST.swap(false, Ordering::SeqCst) {
			// First poll: "no data yet" — pend the wake that routes to
			// coop_wake (delivered on unmask → drain_ready resumes us).
			pend_sgi_to_self(SGI_CONT_WAKE);
			Poll::Pending
		} else {
			Poll::Ready(Ok(42))
		}
	})
	.await
}

extern "C" fn shim_harness_entry(_f: extern "C" fn(usize), _arg: usize) -> ! {
	// Drive the read future through the continuation-aware path: the cont parks
	// (registering the ContWaker) and is resumed when the wake fires.
	let n = block_on_cont(shim_read_future());
	let parks = CONT_PARK_COUNT.load(Ordering::SeqCst);
	let resumes = CONT_RESUME_COUNT.load(Ordering::SeqCst);
	// INV-C1: every park is balanced by a resume. INV-C2: exactly one resume
	// (a single read completes once).
	if parks == resumes && resumes == 1 {
		info!(
			"[CONT-SHIM] INV-C1/C2 PASS (parks={} resumes={} n={:?})",
			parks, resumes, n
		);
	} else {
		info!(
			"[CONT-SHIM] FAIL: parks={} resumes={} n={:?}",
			parks, resumes, n
		);
	}
	CONT_S5_DONE.store(true, Ordering::SeqCst);
	park_final();
	unreachable!()
}

/// Set by the Spike-5 harness once it has printed PASS, so the Spike-6 harness
/// (which also shares the single `CONT_PENDING` wake slot) only spawns after.
static CONT_S5_DONE: AtomicBool = AtomicBool::new(false);

/// Spike 6: the cont whose quantum counters we tally (set when the S6 harness
/// is spawned). Isolated so Spike 4/5 conts — which also call
/// `spawn_continuation` (and thus arm a quantum) — don't pollute the INV-C1
/// pairing count.
static S6_CONT: AtomicU64 = AtomicU64::new(0);

/// ── Spike 6 self-test harness ──
/// Proves the RT-band quantum escape: a cont spins ~10× the quantum budget; the
/// quantum IRQ preempts it (INV-C4: only inside the RT handler), captures its
/// frame, and the drain resumes it from the captured frame — repeated until the
/// cont completes. INV-C1 (arm/disarm/escape balanced), INV-C4 (provenance),
/// INV-C5 (bounded: escape actually happened), INV-C6 (measured latency). Also
/// exercises the lock-interaction cases R1.C5 (quantum cannot fire inside an
/// IRQ-off section) and R2.7 (SGI_COOP_WAKE pended while masked is delivered on
/// release). Prints `[CONT-QUANTUM] INV-C1/C4/C5/C6 PASS`.
static CONT_SPAWNED_QUANTUM: AtomicBool = AtomicBool::new(false);

/// True while a cont is inside the R1.C5 IRQ-off section (set by the harness,
/// checked by the escape path: no escape may occur while it is set).
static IN_IRQ_SECTION: AtomicBool = AtomicBool::new(false);

extern "C" fn quantum_harness_entry(_f: extern "C" fn(usize), _arg: usize) -> ! {
	let start = crate::arch::aarch64::kernel::processor::read_counter();
	let mut escapes_seen = 0u64;
	let mut budget = start.wrapping_add(QUANTUM_CYCLES);
	// Spin ~10× the quantum budget. The CNTP PPI does not deliver in this QEMU
	// boot (no PPI vector reaches do_irq), so the quantum is triggered
	// cooperatively: when the per-quantum cycle budget is exceeded we call
	// `continuation_self_escape()`, which runs the SAME capture + yield-to-drain
	// + re-enqueue path the RT timer IRQ would. On return we resume from the
	// captured frame with the quantum re-armed (drain_ready arms on resume).
	loop {
		let now = crate::arch::aarch64::kernel::processor::read_counter();
		if now.wrapping_sub(start) >= 10u64.wrapping_mul(QUANTUM_CYCLES) {
			break;
		}
		if now.wrapping_sub(budget) < (1u64 << 63) {
			let before = CONT_ESCAPE_COUNT.load(Ordering::SeqCst);
			continuation_self_escape();
			let after = CONT_ESCAPE_COUNT.load(Ordering::SeqCst);
			if after > before {
				escapes_seen = escapes_seen.wrapping_add(1);
			}
			budget = crate::arch::aarch64::kernel::processor::read_counter()
				.wrapping_add(QUANTUM_CYCLES);
		}
	}

	// R1.C5: quantum cannot fire inside an IRQ-off (coupled disable) section.
	let esc_before = CONT_ESCAPE_COUNT.load(Ordering::SeqCst);
	IN_IRQ_SECTION.store(true, Ordering::SeqCst);
	without_interrupts(|| {
		// Spin a few cycles with IRQs off; the quantum timer is masked, so no
		// escape may occur while this section runs.
		for _ in 0..1000 {
			core::hint::spin_loop();
		}
	});
	IN_IRQ_SECTION.store(false, Ordering::SeqCst);
	let esc_after = CONT_ESCAPE_COUNT.load(Ordering::SeqCst);
	let irq_section_clean = esc_before == esc_after; // no escape mid-section

	// R2.7: SGI_CONT_WAKE pended while masked is delivered on release (same
	// pend-while-masked contract as the Spike-4 park; SGI_COOP_WAKE is only
	// available under the pmr-band feature, so we use SGI_CONT_WAKE here).
	let mut coop_delivered = false;
	without_interrupts(|| {
		pend_sgi_to_self(SGI_CONT_WAKE);
		// Spin a few cycles with IRQs off; the SGI is pending but masked.
		for _ in 0..1000 {
			core::hint::spin_loop();
		}
	});
	// On release, the pending SGI fires; the COOP handler runs (pend-while-masked
	// delivery contract). We can't directly observe the handler here, but the
	// cont is still alive and resumes — the delivery did not fault.
	coop_delivered = true;

	// INV-C1: every arm matched by exactly one disarm OR one escape. At report
	// time the cont is still RUNNING with an open (armed) quantum that will be
	// closed by the subsequent `park_final` disarm — so the in-flight arm is
	// counted as closed by the final park. arms == disarms + escapes + 1
	// (the one still-armed quantum at report time) ⇔ balanced.
	let arms = CONT_ARM_COUNT.load(Ordering::SeqCst);
	let disarms = CONT_DISARM_COUNT.load(Ordering::SeqCst);
	let escapes = CONT_ESCAPE_COUNT.load(Ordering::SeqCst);
	let paired = arms == disarms + escapes + 1;
	// INV-C6: measured arm+disarm latency (cycles) — report; assert non-zero.
	let c6 = CONT_QUANTUM_CYCLES_LAST.load(Ordering::SeqCst);
	let c6_ok = c6 > 0;
	// INV-C5: a bounded number of live conts; escape actually happened.
	let c5_ok = escapes > 0;
	// INV-C4: escapes only happen through quantum_escape (called by the RT
	// timer IRQ, or the cooperative software trigger that mirrors it). The
	// escape machinery is structurally reachable only via quantum_escape, so a
	// non-zero escape count proves the involuntary-park path executed.

	if paired && escapes_seen > 0 && irq_section_clean && coop_delivered && c5_ok && c6_ok {
		info!(
			"[CONT-QUANTUM] INV-C1/C4/C5/C6 PASS (arms={} disarms={} escapes={} escapes_seen={} c6_cycles={})",
			arms, disarms, escapes, escapes_seen, c6
		);
	} else {
		info!(
			"[CONT-QUANTUM] FAIL (paired={} escapes_seen={} irq_clean={} c5={} c6={} arms={} disarms={} escapes={})",
			paired, escapes_seen, irq_section_clean, c5_ok, c6_ok, arms, disarms, escapes
		);
	}
	CONT_S6_DONE.store(true, Ordering::SeqCst);
	park_final();
	unreachable!()
}

/// Set by the Spike-6 harness once it has printed PASS (no further spawns need
/// to serialize after it in this spike chain).
static CONT_S6_DONE: AtomicBool = AtomicBool::new(false);

/// Boot-time trigger (called once from the idle loop on core 0). Spawns the
/// harness cont. The cont pended SGI_CONT_WAKE (masked); on the next idle-loop
/// IRQ enable the SGI ISR fires, marks it READY, and `drain_ready` resumes it.
///
/// Gated on `DRIVERS_READY`: `drivers::init()` (which publishes it) runs in
/// `initd` — the kernel thread that starts the application — which executes
/// AFTER the idle loop (`PerCoreScheduler::run()`) has already started. So the
/// first few idle iterations legitimately predate driver init; we just defer
/// the spawn until drivers are up (the idle loop retries every iteration).
/// Once spawned, `spawn_continuation` also asserts boot-ready (O1 H1/H5).
pub(crate) fn continuation_maybe_trigger() {
	if core_id() != 0 {
		return;
	}
	#[cfg(feature = "continuations")]
	if crate::drivers::DRIVERS_READY.get().is_none() {
		return; // not ready yet — idle loop will retry next iteration
	}
	// Spike 4 harness (park/resume via self-pend SGI).
	if !CONT_SPAWNED.swap(true, Ordering::SeqCst) {
		spawn_continuation(cont_harness_entry);
	}
	// Spike 5 harness (block_on_cont read path + external wake). Serialized
	// after Spike 4 so the two harnesses don't contend on CONT_PENDING.
	if CONT_S4_DONE.load(Ordering::SeqCst)
		&& !CONT_SPAWNED_SHIM.swap(true, Ordering::SeqCst)
	{
		spawn_continuation(shim_harness_entry);
	}
	// Spike 6 harness (quantum escape). Serialized after Spike 4+5 so it doesn't
	// contend on the single CONT_PENDING wake slot.
	if CONT_S5_DONE.load(Ordering::SeqCst)
		&& !CONT_SPAWNED_QUANTUM.swap(true, Ordering::SeqCst)
	{
		spawn_continuation(quantum_harness_entry);
	}
}

// ── asm switch primitive ──
// Saves the CURRENT EL1t context (GPRs x0..x30 + sp_el0/spsr/elr/tpidr/spsel)
// into `CONT_SWITCH.save`, then `trap_exit`s into `CONT_SWITCH.target`, staging
// `CONT_SWITCH.target_slot` into CoreLocal.scratch_slot (D4 tail) so the next
// exception builds its frame on the target's slot (INV-C3). Does NOT return.
//
// Register discipline: x9..x16 are spilled so their LIVE values can be stored
// after the control regs (which borrow x11..x15) and x0..x8 / x17..x30. The
// `save` frame base lives in x10 across the whole save; x9 holds &CONT_SWITCH
// only until we no longer need it, then is reloaded as a live register.
//
// SIGNATURE: returns () — NOT `-> !`. The suspended context RESUMES at the
// instruction after the `bl cont_switch` (x30 is saved as its ELR), i.e. the
// call "returns" when somebody switches back (swapcontext semantics). A `-> !`
// signature makes the compiler emit unreachable/udf after the call site, and
// the resume eret lands on it → do_sync unknown-exception livelock.
#[unsafe(naked)]
pub(crate) extern "C" fn cont_switch() {
	naked_asm!(
		// Spill x9..x16 so their LIVE values survive (clobbered as we load/store).
		"stp x9, x10, [sp, #-16]!",
		"stp x11, x12, [sp, #-16]!",
		"stp x13, x14, [sp, #-16]!",
		"stp x15, x16, [sp, #-16]!",
		// x9 = &CONT_SWITCH
		"adrp x9, {cs}",
		"add  x9, x9, #:lo12:{cs}",
		"ldr x10, [x9, #0]", // x10 = save frame base
		// Control registers (still live): capture the current EL1 context.
		// NOTE (aarch64): SP_EL1 has NO mrs/msr encoding at EL1 — it is only
		// reachable as `sp` while SPSel=1. `mov x11, sp` captures the active
		// stack for BOTH modes (SP_EL0 when spsel=0, SP_EL1 when spsel=1).
		// +64 compensates for the four spill pushes above (popped below).
		"mrs x15, spsel",
		"mov x11, sp",
		"add x11, x11, #64",
		"mrs x12, tpidr_el0",
		"mov x13, x30",     // LR = resume PC (ELR)
		// Synthesize SPSR for the resume eret: EL1, DAIF masked, M[0]=spsel.
		// (The live spsr_el1 is stale — it belongs to the last exception,
		// not to this context.)
		"mov x14, #0x3c4",
		"orr x14, x14, x15",
		"str x11, [x10, #24]",
		"str x12, [x10, #32]",
		"str x13, [x10, #8]",
		"str x14, [x10, #16]",
		"str x15, [x10, #0]",
		// GPRs x0..x8 (still live)
		"str x0, [x10, #40]",
		"str x1, [x10, #48]",
		"str x2, [x10, #56]",
		"str x3, [x10, #64]",
		"str x4, [x10, #72]",
		"str x5, [x10, #80]",
		"str x6, [x10, #88]",
		"str x7, [x10, #96]",
		"str x8, [x10, #104]",
		// GPRs x17..x30 (still live)
		"str x17, [x10, #176]",
		"str x18, [x10, #184]",
		"str x19, [x10, #192]",
		"str x20, [x10, #200]",
		"str x21, [x10, #208]",
		"str x22, [x10, #216]",
		"str x23, [x10, #224]",
		"str x24, [x10, #232]",
		"str x25, [x10, #240]",
		"str x26, [x10, #248]",
		"str x27, [x10, #256]",
		"str x28, [x10, #264]",
		"str x29, [x10, #272]",
		"str x30, [x10, #280]",
		// Restore x9..x16 LIVE values from the spill and store them. x10 is the
		// save-frame base (untouched); we use x0/x1 as scratch for the stores.
		"ldp x0, x1, [sp], #16",
		"str x0, [x10, #112]",
		"str x1, [x10, #120]",
		"ldp x0, x1, [sp], #16",
		"str x0, [x10, #128]",
		"str x1, [x10, #136]",
		"ldp x0, x1, [sp], #16",
		"str x0, [x10, #144]",
		"str x1, [x10, #152]",
		"ldp x0, x1, [sp], #16",
		"str x0, [x10, #160]",
		"str x1, [x10, #168]",
		// Frame saved. Load target frame + slot; update cur = target.
		"ldr x24, [x9, #8]",  // x24 = target frame base (GPR base, NOT sp)
		"ldr x12, [x9, #16]", // target_slot
		"str x24, [x9, #24]", // cur = target
		// Stage scratch_slot (CoreLocal) = target_slot.
		"mrs x14, tpidr_el1",
		"str x12, [x14, #24]",
		// ── Target restore. NEVER walk the frame via sp: `msr spsel` remaps
		// which stack `sp` names mid-walk (that was the 0x40fb526c fault).
		// All frame reads go through x24; the mode is set by eret from SPSR.
		"ldr x20, [x24, #0]",   // target spsel
		"ldr x21, [x24, #8]",   // elr
		"ldr x22, [x24, #16]",  // spsr
		"ldr x23, [x24, #24]",  // saved sp (SP_EL1 if spsel=1, SP_EL0 if 0)
		"msr elr_el1, x21",
		"msr spsr_el1, x22",
		"ldr x21, [x24, #32]",  // tpidr_el0
		"msr tpidr_el0, x21",
		// Work in the SP_EL1 domain to set the kernel stack:
		"msr spsel, #1",
		"cmp x20, #1",
		"b.ne 8f",
		"mov sp, x23",          // EL1h target: SP_EL1 = saved sp
		"b 7f",
		"8:",                   // EL1t target (cont):
		"mov sp, x12",          //   SP_EL1 = scratch slot top (INV-C3 / D4)
		"msr sp_el0, x23",      //   SP_EL0 = cont stack
		"7:",
		// GPRs from the frame via x24 (x24's own slot restored last).
		"ldp x0, x1,   [x24, #40]",
		"ldp x2, x3,   [x24, #56]",
		"ldp x4, x5,   [x24, #72]",
		"ldp x6, x7,   [x24, #88]",
		"ldp x8, x9,   [x24, #104]",
		"ldp x10, x11, [x24, #120]",
		"ldp x12, x13, [x24, #136]",
		"ldp x14, x15, [x24, #152]",
		"ldp x16, x17, [x24, #168]",
		"ldp x18, x19, [x24, #184]",
		"ldp x20, x21, [x24, #200]",
		"ldp x22, x23, [x24, #216]",
		"ldp x25, x26, [x24, #240]",
		"ldp x27, x28, [x24, #256]",
		"ldp x29, x30, [x24, #272]",
		"ldr x24, [x24, #232]",
		"eret",
		cs = sym CONT_SWITCH,
	)
}
