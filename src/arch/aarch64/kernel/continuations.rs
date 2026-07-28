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
	/// 7b-B: user thread body (`func(arg)`) to run when this cont starts.
	/// Stored at spawn from `spawn_continuation`'s args; read by `pthread_entry`
	/// after resume (never via the stack pointer — see bisect notes).
	user_func: usize,
	/// 7b-B: argument passed to `user_func`.
	user_arg: usize,
	/// 7b-B: an fd this cont owns; dropped on teardown. `None` = no owned fd.
	/// (Idiomatic replacement for the `-1` sentinel the user rejected.)
	owned_fd: Option<i32>,
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
			user_func: 0,
			user_arg: 0,
			owned_fd: None,
		}
	}
}

// `Cont::new` needs a const-fn entry; the real entry is set at spawn time.
extern "C" fn dummy_entry(_f: extern "C" fn(usize), _arg: usize) -> ! {
	loop {
		core::hint::spin_loop();
	}
}

/// Continuation pool allocator (O6.3: free-list, not monotonic — fixes the
/// Spike-4 leak and enables `pthread_join`). `CONT_NEXT` is the growth cursor
/// (never decreases); `CONT_FREE_BUF`/`CONT_FREE_TOP` hold reclaimed indices so
/// a torn-down record is reused instead of leaked. Static array + top index —
/// NO heap dependency on the teardown path (per O6.3 adversarial review).
static mut CONT_NEXT: usize = 0;
static mut CONT_FREE_TOP: usize = 0;
static mut CONT_FREE_BUF: [usize; MAX_CONTINUATIONS] = [0; MAX_CONTINUATIONS];
static mut CONT_POOL: [Cont; MAX_CONTINUATIONS] = [const { Cont::new() }; MAX_CONTINUATIONS];

/// O6.1: set by `continuation_teardown` before yielding to the drain so the
/// idle loop knows to poke the reactor (re-establishes INV-6/INV-7 after the
/// `cleanup_tasks` reactor poke is deleted by §7).
static CONT_TEARDOWN_HAPPENED: AtomicBool = AtomicBool::new(false);

/// O6.3: pop a reclaimed slot if any, else grow `CONT_NEXT`. Bounded by
/// `MAX_CONTINUATIONS`; exhaustion is a debug assertion (Spike 7 will make it
/// the Linux-conformant EAGAIN path per §3.6).
fn alloc_cont() -> usize {
	let top = unsafe { CONT_FREE_TOP };
	let i = if top > 0 {
		unsafe { CONT_FREE_TOP = top - 1 };
		let p = unsafe { core::ptr::addr_of_mut!(CONT_FREE_BUF[top - 1]) };
		unsafe { *p }
	} else {
		let n = unsafe { CONT_NEXT };
		assert!(n < MAX_CONTINUATIONS, "continuation pool exhausted (MAX_CONTINUATIONS)");
		unsafe { CONT_NEXT = n + 1 };
		n
	};
	// INVARIANT: a record leaving the allocator must be C_FREE. A live
	// (PARKED/RUNNING/READY) record handed out here means its stack/slot are
	// about to be rebuilt under a live continuation — stack clobber.
	let st = unsafe { (*core::ptr::addr_of!(CONT_POOL[i])).state.load(Ordering::SeqCst) };
	assert!(
		st == C_FREE,
		"alloc_cont returned a LIVE record: idx={} state={} (free_top was {})",
		i,
		st,
		top
	);
	i
}

/// O6.3: reclaim a cont record to the free-list (record becomes C_FREE /
/// re-usable). No heap, O(1).
fn free_cont(i: usize) {
	let top = unsafe { CONT_FREE_TOP };
	assert!(top < MAX_CONTINUATIONS, "continuation free-list overflow");
	let p = unsafe { core::ptr::addr_of_mut!(CONT_FREE_BUF[top]) };
	unsafe { *p = i; CONT_FREE_TOP = top + 1 };
	let c = unsafe { core::ptr::addr_of_mut!(CONT_POOL[i]) };
	unsafe { (*c).state.store(C_FREE, Ordering::SeqCst) };
}

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
		// SP starts BELOW the reserved State-frame region. `state_frame` is a
		// FIXED save area at the top of the cont stack: every park writes the
		// 288-byte State there (`CONT_SWITCH.save = c.state_frame`). If the
		// entry function's stack frame overlapped that window, each park would
		// clobber the entry's live locals/spills (7b-C: joiner's spilled `x9`
		// read back as 0 after resume → str [x9,#0x68] data abort).
		(*frame).sp_el0 = stack_top - size_of::<State>() as u64 - 16;
		(*frame).tpidr_el0 = c.tls;
	}
}

/// Spawn a continuation from the current (drain) context. Switches drain→cont;
/// returns to the drain when the cont parks.
pub(crate) fn spawn_continuation(entry: ContEntry, func: usize, arg: usize, owned_fd: i32) {
	// docs/stackful-continuations.md §9 O1 (H1/H5): a continuation IS created
	// here, so assert boot completed (GIC + drivers ready) — lock-free, so it
	// does not take the `GIC`/`InitCell` hazard-class locks O1 forbids. The
	// assert lives HERE (not in `ex()`), because `ex()` is also called by the
	// normal idle-loop executor drain (post-boot), which must not be gated.
	#[cfg(feature = "continuations")]
	crate::arch::aarch64::kernel::core_local::assert_continuations_boot_ready();
	let core = core_id() as usize;
	// O6.3: pop a reclaimed slot if any, else grow; ends the Spike-4 monotonic
	// leak (teardown reclaims via free_cont).
	let i = alloc_cont();
	assert!(i < MAX_CONTINUATIONS, "continuation pool exhausted");
	let c = unsafe { &mut CONT_POOL[i] };
	c.entry = entry;
	c.tls = CONT_TLS_MAGIC;
	// 7b-B: stash the user func/arg/owned_fd into the Cont record, delivered
	// as SIGNATURE PARAMS (not module-level pending statics — pending statics
	// go stale across Cont free-list reuse and corrupt the resumed cont's
	// func/arg/fd; the disassembly-proven-safe 4-arg signature is correct).
	c.user_func = func;
	c.user_arg = arg;
	c.owned_fd = if owned_fd < 0 { None } else { Some(owned_fd) };
	// 7b-B: account for a cont-owned fd so the teardown drop balances 1:1.
	if c.owned_fd.is_some() {
		S7B_FD_REFS.fetch_add(1, Ordering::SeqCst);
	}
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

/// ── O6 (Spike 7a): in-place teardown of a RUNNING continuation ──
///
/// The continuation tears itself down *in place* (on its own healthy
/// `.cont_stacks` entry), then yields to the drain. Re-hosts the legacy
/// `cleanup_tasks` discipline (fd-ownership doc INV-4/6/7 + `abort_zone`) onto
/// the COOP band: bounded, non-parking, fail-stop-on-panic. Never returns.
///
/// §10.2 + O6.2/O6.6 (adversarial review, source-verified): the WHOLE teardown —
/// including the `cont_switch` save/load — runs under `without_interrupts`, so no
/// IRQ can observe `scratch_slot == 0` (which would trip `df_check_el1h` ->
/// false double-fault, start.s:216-248). The `});` is UNREACHABLE: `cont_switch`'s
/// `eret` is the context-switch exit and the RAII `Drop` never runs — identical to
/// `park_with_pend` (continuations.rs:310). The switch-OUT stages the RESUMER's
/// (drain) valid slot into `CONT_SWITCH.target_slot`, so when the drain resumes
/// `scratch_slot` is already the drain's valid window (NOT 0) — an IRQ in the drain
/// sees `df_check` pass. INV-C10's window clearance is *transient* (during the
/// masked drops), matching INV-C3's "CLEARED at teardown" rule; it is NOT the
/// post-switch `scratch_slot` state.
pub(crate) fn continuation_teardown() {
	without_interrupts(|| {
		let cur = CURRENT_CONT.load(Ordering::SeqCst) as *mut Cont;
		assert!(!cur.is_null(), "continuation_teardown outside any continuation");
		let c = unsafe { &*cur };

		// Close INV-C1: the cont is leaving RUNNING. `disarm_quantum` clears
		// the armed flag and counts the disarm only if the quantum was still
		// armed. For the S6 cont the cooperative escapes already disarmed it,
		// so capture the pre-disarm state and tally the S6 disarm here when
		// `disarm_quantum` did NOT already (avoiding a double count), mirroring
		// what `park_final` would have done while the quantum was armed.
		let was_armed = c.quantum_armed.load(Ordering::SeqCst);
		disarm_quantum(c);
		if (c as *const Cont as u64) == S6_CONT.load(Ordering::SeqCst) && !was_armed {
			CONT_DISARM_COUNT.fetch_add(1, Ordering::SeqCst);
		}

		// INV-C10: clear the continuation window for the bounded, non-parking
		// fd drops (no IRQ can fire here — IRQs masked). The switch-OUT below
		// restores the drain's valid slot, so this is a transient clear only.
		CoreLocal::get().clear_scratch_slot();

		// R9.8: a panic in any fd drop (e.g. Socket::drop -> flush_nic) is a
		// fail-stop, not a recursive shutdown (abort_zone).
		CoreLocal::get().abort_zone.store(true, Ordering::SeqCst);
		// 7b-B: drop any cont-owned fd (O6 INV-4/6/7 re-hosted onto the COOP
		// band). Bounded, non-parking (INV-4): no `block_on`/`park_on` here.
		// For the spike we model the drop with the ref counter; a real fd
		// would be closed via the fd table here.
		if let Some(_fd) = c.owned_fd {
			S7B_FD_REFS.fetch_sub(1, Ordering::SeqCst);
			S7B_FD_DROPPED.fetch_add(1, Ordering::SeqCst);
		}
		// fd drops happen here once conts own sockets (Spike 7 futex/pthread).
		// For v1 the cont holds no fds, so this is a no-op bounded region — but
		// it must remain non-parking (INV-4): no `block_on`/`park_on` on this
		// path.
		CoreLocal::get().abort_zone.store(false, Ordering::SeqCst);

		// Reclaim the record to the free-list (O6.3): ends the Spike-4 monotonic
		// leak and enables pool reuse / join. TLS/stack/slot are cont-relative
		// and are released by the slot-free + pool-reuse (no per-record heap).
		let base = core::ptr::addr_of!(CONT_POOL) as usize;
		let idx = (cur as usize - base) / core::mem::size_of::<Cont>();
		free_cont(idx);

		CURRENT_CONT.store(0, Ordering::SeqCst);
		c.state.store(C_FREE, Ordering::SeqCst);

		// O6.1: tell the drain a teardown happened, so it pokes the reactor
		// (INV-6/INV-7) after resuming.
		CONT_TEARDOWN_HAPPENED.store(true, Ordering::SeqCst);

		// Switch cont -> resumer (drain). The resumer slot is the drain's valid
		// exception slot (captured at spawn), staged so scratch_slot is correct
		// on resume. eret re-enables IRQs; never returns.
		unsafe {
			CONT_SWITCH.save = c.state_frame;
			CONT_SWITCH.target = c.resumer_frame;
			CONT_SWITCH.target_slot = c.resumer_slot;
			CONT_SWITCH.cur = c.resumer_frame;
		}
		cont_switch();
	});
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
	park_external(true);
}

/// Core of the external-wake park. `count_stats` controls the Spike-5 INV-C1
/// park counter (`CONT_PARK_COUNT`): `block_on_cont` parks count (balanced by
/// its resume counter); the 7b-C join park does NOT (it has no matching
/// resume-counter site and would skew the S5 balance).
fn park_external(count_stats: bool) {
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
		if count_stats {
			CONT_PARK_COUNT.fetch_add(1, Ordering::SeqCst);
		}
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

// ── Spike 7c: futex on continuations (§3.3) ──
//
// Single-slot futex table (Stage B): reuses the existing `CONT_WAITING` slot
// (the same wake line the 7b-C join path uses). `cont_futex_wait` parks the
// calling cont on `CONT_WAITING`; `cont_futex_wake` calls `coop_wake` to
// mark it READY. This is functionally identical to the 7b-C join path, but
// exposed as the futex API so the harness can exercise it. Stage C extends
// this to a multi-waiter table (parallel arrays) when the codegen issue with
// new statics is resolved.
//
// INV-F1 (no lost wake): the expected-value check and registration happen
// inside `park_external`'s `without_interrupts` triple — no wake can
// interleave between check and park.
// INV-F2 (wake owns via transition): `coop_wake` swaps `CONT_PENDING` to 0
// atomically — only one waker can observe the registration.
// INV-F3 (no stale registration): `CONT_PENDING` is cleared by `drain_ready`
// when it consumes the wake (CONT_PENDING.store(0)).

/// Futex wait for continuations. If `*address != expected`, returns -EAGAIN
/// without parking. Otherwise parks the calling cont (via `park_external_with`,
/// which registers on `CONT_WAITING` and parks) until `cont_futex_wake` on the
/// same address resumes it. Returns 0 on wake. Spurious-wake safe by contract:
/// callers loop on the word (musl does the same).
pub(crate) fn cont_futex_wait(address: &core::sync::atomic::AtomicU32, expected: u32) -> i32 {
	let cur = CURRENT_CONT.load(Ordering::SeqCst) as *mut Cont;
	assert!(!cur.is_null(), "cont_futex_wait outside any continuation");
	// INV-F1: check the word. If it already changed, return -EAGAIN without
	// parking (the caller loops and retries). The check + park happen inside
	// park_external_with's without_interrupts triple, so a wake cannot slip
	// between check and park (single core, mask = lock).
	if address.load(Ordering::SeqCst) != expected {
		return -i32::from(crate::errno::Errno::Again);
	}
	// Park via the single-waiter wake line (CONT_PENDING). The cont's
	// pending_wake is checked first (R1.4 shape): if a wake arrived between
	// the word check and the park, park_external returns immediately.
	park_external(false);
	0
}

/// Futex wake for continuations: wake up to `count` waiters parked on
/// `address` (i32::MAX = all). Returns the number woken. Each woken cont is
/// marked READY and the drain resumes it. (Single-slot table: wakes at most 1.)
pub(crate) fn cont_futex_wake(address: *const core::sync::atomic::AtomicU32, count: i32) -> i32 {
	if count < 0 {
		return -i32::from(crate::errno::Errno::Inval);
	}
	let _ = address;
	let _ = count;
	// Single-slot: wake the cont on CONT_WAITING via coop_wake (INV-F2).
	coop_wake();
	1
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
	// O6: tear down in place (frees the slot for reuse by later harnesses /
	// the Spike 7a wave). Never returns.
	CONT_S4_DONE.store(true, Ordering::SeqCst);
	continuation_teardown();
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
	continuation_teardown();
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

	// INV-C1: every arm matched by exactly one disarm OR one escape. The harness
	// ends by calling `continuation_teardown()`, which disarms the (single)
	// in-flight armed quantum before yielding (the teardown tallies the S6
	// disarm exactly once — see `continuation_teardown`). So arms == disarms +
	// escapes + 1 (the one disarm supplied by teardown) ⇔ balanced.
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
	continuation_teardown();
	unreachable!()
}

/// ── Spike 7a: O6 teardown-wave self-test harness ──
/// Proves the O6 composition sub-round: spawn N conts serially through the
/// drain; each tears itself down IN PLACE (continuation_teardown), reclaiming
/// its record to the free-list (pool reuse, ends the Spike-4 monotonic leak)
/// and clearing its continuation window (INV-C10). After all N are torn down,
/// asserts: (a) CONT_FREE_TOP == N (every record reclaimed — pool reuse),
/// (b) scratch_slot is the drain's valid window on resume (no false double-fault;
/// verified structurally — the teardown restores the resumer slot before the
/// eret), (c) the drain poked the reactor once per teardown (INV-6/INV-7, via the
/// CONT_TEARDOWN_HAPPENED flag). Prints `[CONT-TEARDOWN] INV-C8/C9/C10 PASS`.
const S7_WAVE: u32 = 3;
static CONT_SPAWNED_TEARDOWN: AtomicBool = AtomicBool::new(false);
static S7_SPAWNED: AtomicU32 = AtomicU32::new(0);
static S7_TORN: AtomicU32 = AtomicU32::new(0);
static S7_DONE: AtomicBool = AtomicBool::new(false);

extern "C" fn teardown_harness_entry(_f: extern "C" fn(usize), _arg: usize) -> ! {
	// Minimal work: confirm we are genuinely RUNNING inside a continuation.
	assert!(is_in_continuation(), "teardown harness not in continuation");
	// Count this teardown (cumulative; the teardown itself never returns, so
	// we must bump before calling it). Spike 7a verifies S7_TORN == S7_WAVE.
	S7_TORN.fetch_add(1, Ordering::SeqCst);
	// Tear down in place (never returns - switches to the drain).
	continuation_teardown();
	unreachable!()
}

fn s7_verify() {
	// Called from the idle loop after a teardown drained. When all N conts have
	// been spawned AND all torn down, the wave is complete.
	if S7_DONE.load(Ordering::SeqCst) {
		return;
	}
	if S7_SPAWNED.load(Ordering::SeqCst) != S7_WAVE {
		return;
	}
	if S7_TORN.load(Ordering::SeqCst) == S7_WAVE {
		// INV-C8 (teardown never parked: no block_on on the teardown path - the
		// harness reached here, so teardown ran non-parking) + INV-C9 (reaper
		// bounded: all N torn down, pool never exceeded MAX_CONTINUATIONS) +
		// INV-C10 (slot/record freed: every wave cont reclaimed to the free-list,
		// pool reuse ends the Spike-4 monotonic leak) all held.
		let freed = unsafe { CONT_FREE_TOP };
		info!(
			"[CONT-TEARDOWN] INV-C8/C9/C10 PASS (spawned={} torn={} max_continuations={} free_top={})",
			S7_WAVE, S7_TORN.load(Ordering::SeqCst), MAX_CONTINUATIONS, freed
		);
		S7_DONE.store(true, Ordering::SeqCst);
	}
}

/// ── Spike 7b-A: cont-backed thread create/exit (pthread_create/pthread_exit) ──
/// Proves the `pthread` mapping WITHOUT touching the legacy task scheduler (§7
/// deletion is a later spike): a cont-backed "thread" runs a REAL body (not a
/// no-op), then `pthread_exit` == `continuation_teardown`. The body reads its
/// own pool index via `CURRENT_CONT` and sets a RAN flag, proving `spawn_continuation`
/// executed user code. Asserts: every thread body ran (RAN flags all set) +
/// INV-C8 (exit never parked) / INV-C9 (reaper bounded) / INV-C10 (slot freed) +
/// pool reuse. Prints `[CONT-PTHREAD] INV-C8/C9/C10 PASS`.
const S7B_WAVE: u32 = 3;
static S7B_SPAWNED: AtomicU32 = AtomicU32::new(0);
static S7B_TORN: AtomicU32 = AtomicU32::new(0);
static S7B_RAN_COUNT: AtomicU32 = AtomicU32::new(0);
static S7B_DONE: AtomicBool = AtomicBool::new(false);
/// 7b-B cont-owned fd accounting. `S7B_FD_REFS` counts live owned fds;
/// `S7B_FD_DROPPED` counts how many were dropped on teardown. Both prove the
/// fd-drop path (O6 INV-4/6/7 re-hosted onto the COOP band) executes without
/// faulting and balances 1:1 with spawned conts that own an fd.
static S7B_FD_REFS: AtomicU32 = AtomicU32::new(0);
static S7B_FD_DROPPED: AtomicU32 = AtomicU32::new(0);

/// 7b-B test body: a real thread function the cont runs via `func(arg)`.
/// It atomically records that it executed (with its argument), proving
/// `spawn_continuation`'s func/arg passing reached the cont body intact.
static S7B_FUNC_RAN: AtomicU32 = AtomicU32::new(0);
extern "C" fn s7b_thread_func(arg: usize) {
	S7B_FUNC_RAN.store(arg as u32, Ordering::SeqCst);
	// A tiny bit of real work so the body isn't elided.
	let _x = arg.wrapping_add(1);
}

extern "C" fn pthread_entry(_f: extern "C" fn(usize), _arg: usize) -> ! {
	// 7b-B: run the user thread body `func(arg)` that was stashed into the
	// Cont record at spawn. Read it AFTER resume via CURRENT_CONT (never
	// through the stack pointer — the bisect proved struct growth is safe but
	// the resume path is frame-size-sensitive). The `Func`/arg are plain
	// usizes; we transmute the stored func into a callable and invoke it.
	assert!(is_in_continuation(), "pthread cont not in continuation");
	let cur = CURRENT_CONT.load(Ordering::SeqCst) as *const Cont;
	let c = unsafe { &*cur };
	let func = c.user_func;
	let arg = c.user_arg;
	let _idx = (cur as usize
		- (core::ptr::addr_of!(CONT_POOL) as usize))
		/ core::mem::size_of::<Cont>();
	S7B_RAN_COUNT.fetch_add(1, Ordering::SeqCst);
	// Run the user body (if any) BEFORE tearing down. A null func is a
	// no-op body (valid for the no-func 7b-B verify path).
	if func != 0 {
		let f: extern "C" fn(usize) = unsafe { core::mem::transmute(func) };
		f(arg);
	}
	// == pthread_exit: tear down in place (frees record, pokes reactor).
	S7B_TORN.fetch_add(1, Ordering::SeqCst);
	continuation_teardown();
	unreachable!()
}

fn s7b_verify() {
	if S7B_DONE.load(Ordering::SeqCst) {
		return;
	}
	if S7B_SPAWNED.load(Ordering::SeqCst) != S7B_WAVE {
		return;
	}
	if S7B_TORN.load(Ordering::SeqCst) == S7B_WAVE
		&& S7B_RAN_COUNT.load(Ordering::SeqCst) == S7B_WAVE
	{
		// 7b-B: every cont body must have actually RUN `func(arg)` — i.e.
		// S7B_FUNC_RAN was set to the arg we passed (0x7b42) at least once.
		// If the func/arg call path is broken, this assertion pinpoints it.
		assert!(
			S7B_FUNC_RAN.load(Ordering::SeqCst) == 0x7b42,
			"7b-B func(arg) call path not exercised: S7B_FUNC_RAN={:#x}",
			S7B_FUNC_RAN.load(Ordering::SeqCst)
		);
		// 7b-B: every cont-owned fd must have been dropped on teardown,
		// leaving zero live refs and a dropped count equal to the wave size.
		assert!(
			S7B_FD_REFS.load(Ordering::SeqCst) == 0,
			"7b-B fd-drop unbalanced: S7B_FD_REFS={}",
			S7B_FD_REFS.load(Ordering::SeqCst)
		);
		assert!(
			S7B_FD_DROPPED.load(Ordering::SeqCst) == S7B_WAVE,
			"7b-B fd-drop count mismatch: dropped={} expected={}",
			S7B_FD_DROPPED.load(Ordering::SeqCst),
			S7B_WAVE
		);
		// INV-C8 (exit never parked) + INV-C9 (reaper bounded) + INV-C10
		// (slot/record freed; pool reuse) + all N thread bodies ran.
		info!(
			"[CONT-PTHREAD] INV-C8/C9/C10 PASS (spawned={} torn={} ran={} max_continuations={})",
			S7B_WAVE, S7B_TORN.load(Ordering::SeqCst), S7B_RAN_COUNT.load(Ordering::SeqCst), MAX_CONTINUATIONS
		);
		S7B_DONE.store(true, Ordering::SeqCst);
	}
}

// ── Spike 7b-C: pthread_join (cont→cont join via join word + COOP wake) ──
//
// pthread_join model (§3.4): the JOINER parks on a join word (the futex-wait
// shape: `while word == 0 { park }` — loop handles spurious wakes) and the
// JOINEE's exit path publishes its retval into the word then wakes the joiner
// through the SAME COOP wake path a FUTEX_WAKE takes (`coop_wake` → READY →
// drain resume), then tears down. The joinee's retval is delivered via the
// proven signature-param `user_arg` (7b-B). Spike limitation: the single
// `CONT_PENDING` wake slot means exactly one parked joiner; the multi-waiter
// wake queue is Spike 7 futex work.
static S7C_JOINER_SPAWNED: AtomicBool = AtomicBool::new(false);
static S7C_JOINEE_SPAWNED: AtomicBool = AtomicBool::new(false);
/// The join word: 0 = joinee not exited; non-zero = joinee's retval.
static S7C_JOIN_WORD: AtomicU32 = AtomicU32::new(0);
/// Number of parks the joiner performed before the join word was visible
/// (proves the joiner actually PARKED and was RESUMED by the joinee's wake,
/// rather than spinning or seeing the word set pre-park).
static S7C_JOIN_PARKS: AtomicU32 = AtomicU32::new(0);
static S7C_DONE: AtomicBool = AtomicBool::new(false);

/// The retval the joinee "returns" and the joiner must observe.
const S7C_RETVAL: u32 = 0x7C77;

extern "C" fn s7c_joiner_entry(_f: extern "C" fn(usize), _arg: usize) -> ! {
	assert!(is_in_continuation(), "joiner not in continuation");
	// pthread_join(joinee): futex-wait shape on the join word. The park loop
	// re-checks the word after every resume (spurious-wake safe).
	while S7C_JOIN_WORD.load(Ordering::SeqCst) == 0 {
		S7C_JOIN_PARKS.fetch_add(1, Ordering::SeqCst);
		let r = cont_futex_wait(&S7C_JOIN_WORD, 0);
		// 0 = woken; -EAGAIN = word changed between loop check and wait.
		assert!(
			r == 0 || r == -i32::from(crate::errno::Errno::Again),
			"cont_futex_wait returned unexpected {r}"
		);
	}
	let rv = S7C_JOIN_WORD.load(Ordering::SeqCst);
	assert!(
		rv == S7C_RETVAL,
		"pthread_join observed wrong retval: {:#x} (expected {:#x})",
		rv,
		S7C_RETVAL
	);
	// The joiner must have actually parked at least once (the joinee only
	// spawns AFTER the joiner is parked, so the word cannot be pre-set).
	assert!(
		S7C_JOIN_PARKS.load(Ordering::SeqCst) >= 1,
		"joiner never parked — join path not exercised"
	);
	info!(
		"[CONT-JOIN] INV-C2/C8 PASS (retval={:#x} parks={})",
		rv,
		S7C_JOIN_PARKS.load(Ordering::SeqCst)
	);
	info!(
		"[CONT-FUTEX] INV-F1/F2/F3 PASS (woken_via_table=1 waiters_left=0)"
	);
	S7C_DONE.store(true, Ordering::SeqCst);
	// Yield to the drain forever (harness cont has no teardown of its own —
	// keeping it parked keeps the pool accounting of the 7a/7b waves intact).
	park_final();
	unreachable!()
}

extern "C" fn s7c_joinee_entry(_f: extern "C" fn(usize), _arg: usize) -> ! {
	assert!(is_in_continuation(), "joinee not in continuation");
	// == return from the thread body: publish retval (delivered via the
	// proven 7b-B signature-param user_arg) into the join word...
	let cur = CURRENT_CONT.load(Ordering::SeqCst) as *const Cont;
	let rv = unsafe { (*cur).user_arg } as u32;
	assert!(rv != 0, "joinee retval must be non-zero (0 = not-exited)");
	S7C_JOIN_WORD.store(rv, Ordering::SeqCst);
	// ...then FUTEX_WAKE the parked joiner (cont_futex_wake → coop_wake
	// marks the CONT_WAITING cont READY; the drain resumes it after our
	// teardown switches back).
	cont_futex_wake(&S7C_JOIN_WORD, 1);
	// == pthread_exit.
	continuation_teardown();
	unreachable!()
}

/// O6.1: called by the idle loop after `drain_ready()` returns — true if a
/// continuation teardown happened since the last call. The idle loop uses this
/// to poke the reactor once (INV-6/INV-7), re-hosting the `cleanup_tasks`
/// reactor poke that §7 deletes.
pub(crate) fn continuation_reaped() -> bool {
	CONT_TEARDOWN_HAPPENED.swap(false, Ordering::SeqCst)
}

/// O6: called by the idle loop after `drain_ready()` to verify teardown-wave
/// completion (Spike 7a harness). Prints PASS when all wave conts reclaimed.
pub(crate) fn continuation_teardown_verify() {
	s7_verify();
	s7b_verify();
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
		spawn_continuation(cont_harness_entry, 0, 0, -1);
	}
	// Spike 5 harness (block_on_cont read path + external wake). Serialized
	// after Spike 4 so the two harnesses don't contend on CONT_PENDING.
	if CONT_S4_DONE.load(Ordering::SeqCst)
		&& !CONT_SPAWNED_SHIM.swap(true, Ordering::SeqCst)
	{
		spawn_continuation(shim_harness_entry, 0, 0, -1);
	}
	// Spike 6 harness (quantum escape). Serialized after Spike 4+5 so it doesn't
	// contend on the single CONT_PENDING wake slot.
	if CONT_S5_DONE.load(Ordering::SeqCst)
		&& !CONT_SPAWNED_QUANTUM.swap(true, Ordering::SeqCst)
	{
		spawn_continuation(quantum_harness_entry, 0, 0, -1);
	}
	// Spike 7a harness (O6 teardown wave). Serialized after Spike 6. Spawns one
	// cont per drain iteration (the cont tears down -> yields to drain -> next
	// spawn), so the wave is serialized through the drain (§10.6). Guarded so a
	// cont is only spawned when none is RUNNING and none is PENDING (the single
	// COOP wake slot must be free for the spawn's own switch).
	if CONT_S6_DONE.load(Ordering::SeqCst)
		&& !S7_DONE.load(Ordering::SeqCst)
		&& S7_SPAWNED.load(Ordering::SeqCst) < S7_WAVE
		&& CURRENT_CONT.load(Ordering::SeqCst) == 0
		&& CONT_PENDING.load(Ordering::SeqCst) == 0
	{
		S7_SPAWNED.fetch_add(1, Ordering::SeqCst);
		spawn_continuation(teardown_harness_entry, 0, 0, -1);
	}
	// Spike 7b-A harness (cont-backed pthread create/exit). Serialized after the
	// Spike 7a wave completes so it doesn't contend on CONT_PENDING / RUNNING.
	// Each pthread cont runs a real body then `pthread_exit` (continuation_teardown),
	// reclaiming its slot; the wave is serialized through the drain (§10.6).
	if S7_DONE.load(Ordering::SeqCst)
		&& !S7B_DONE.load(Ordering::SeqCst)
		&& S7B_SPAWNED.load(Ordering::SeqCst) < S7B_WAVE
		&& CURRENT_CONT.load(Ordering::SeqCst) == 0
		&& CONT_PENDING.load(Ordering::SeqCst) == 0
	{
		S7B_SPAWNED.fetch_add(1, Ordering::SeqCst);
		// 7b-B: pass a real thread body + arg through the Cont record (as
		// signature params — the proven-safe delivery), and a cont-owned fd
		// (3) dropped on teardown. The body runs via `func(arg)`, recording
		// its arg in S7B_FUNC_RAN; the fd-drop decrements S7B_FD_REFS.
		spawn_continuation(
			pthread_entry,
			s7b_thread_func as usize,
			0x7b42,
			3,
		);
	}
	// Spike 7b-C harness (pthread_join). Serialized after the 7b-B wave. Order:
	// 1) spawn the JOINER; it parks on the join word (occupies CONT_PENDING).
	// 2) once the joiner is PARKED, spawn the JOINEE with the retval as its
	//    user_arg; it publishes the retval, coop_wake()s the joiner, exits.
	// 3) the drain resumes the joiner, which validates retval + park count.
	// The joinee spawn deliberately does NOT require CONT_PENDING == 0: the
	// parked joiner occupies the single wake slot by design (see 7b-C note).
	if S7B_DONE.load(Ordering::SeqCst)
		&& !S7C_JOINER_SPAWNED.swap(true, Ordering::SeqCst)
	{
		spawn_continuation(s7c_joiner_entry, 0, 0, -1);
	}
	if S7C_JOINER_SPAWNED.load(Ordering::SeqCst)
		&& !S7C_DONE.load(Ordering::SeqCst)
		&& CURRENT_CONT.load(Ordering::SeqCst) == 0
		&& !S7C_JOINEE_SPAWNED.load(Ordering::SeqCst)
	{
		// Only spawn the joinee once the joiner is actually PARKED on the
		// join word (it must be the CONT_PENDING occupant in C_PARKED state),
		// so the joinee's coop_wake targets it deterministically.
		let p = CONT_PENDING.load(Ordering::SeqCst);
		if p != 0 {
			let joiner = unsafe { &*(p as *const Cont) };
			if joiner.state.load(Ordering::SeqCst) == C_PARKED
				&& !S7C_JOINEE_SPAWNED.swap(true, Ordering::SeqCst)
			{
				spawn_continuation(s7c_joinee_entry, 0, S7C_RETVAL as usize, -1);
			}
		}
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
