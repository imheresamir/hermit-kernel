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
use crate::config::{CONT_GUARD, CONT_SLOT_GUARD, CONT_SLOT_SIZE, CONT_STACK_SIZE, MAX_CONTINUATIONS};

// Continuation lifecycle states (§3.2.1). ESCAPED reserved for Spike 6.
const C_FREE: u32 = 0;
const C_READY: u32 = 1;
const C_RUNNING: u32 = 2;
const C_PARKED: u32 = 3;

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
}

/// Continuation entry signature = `task_start`'s shape, so it stores directly
/// into `State.elr_el1` without a transmute. The cont ignores `f`/`arg`.
pub(crate) type ContEntry = extern "C" fn(extern "C" fn(usize), usize) -> !;

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
		(*frame).spsr_el1 = 0x3e4;
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
	build_frame(c, core, i);
	c.state.store(C_READY, Ordering::SeqCst);

	// resumer = current task (drain) frame + its scratch slot.
	let task_frame = core_scheduler().get_last_stack_pointer().as_u64();
	c.resumer_frame = task_frame as *const State;
	c.resumer_slot = CoreLocal::get().scratch_slot();

	CURRENT_CONT.store(c as *const Cont as u64, Ordering::SeqCst);
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

/// Drain one ready continuation (called from the idle loop on core 0). Switches
/// drain→cont if a cont is READY.
pub(crate) fn drain_ready() {
	let pending = CONT_PENDING.load(Ordering::SeqCst);
	if pending == 0 {
		return;
	}
	let c = unsafe { &*(pending as *const Cont) };
	if c.state.load(Ordering::SeqCst) != C_READY {
		return;
	}
	// This resume CONSUMES the wake that made the cont READY (R1.4): clear
	// pending_wake so a later park doesn't see a stale wake and refuse to park.
	c.pending_wake.store(0, Ordering::SeqCst);
	let task_frame = core_scheduler().get_last_stack_pointer().as_u64();
	unsafe {
		(*(pending as *mut Cont)).resumer_frame = task_frame as *const State;
		(*(pending as *mut Cont)).resumer_slot = CoreLocal::get().scratch_slot();
		CONT_SWITCH.save = task_frame as *const State;
		CONT_SWITCH.target = c.state_frame;
		CONT_SWITCH.target_slot = c.slot_top;
		CONT_SWITCH.cur = c.state_frame;
	}
	c.state.store(C_RUNNING, Ordering::SeqCst);
	CONT_PENDING.store(0, Ordering::SeqCst);
	CURRENT_CONT.store(pending, Ordering::SeqCst);
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
	park_final();
	// `park_final` never returns (cont_switch -> eret to the drain); this is
	// unreachable but makes the `-> !` signature explicit.
	unreachable!()
}

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
	if crate::drivers::DRIVERS_READY.get().is_some() {
		// drivers up; safe to (once) spawn the harness
	} else {
		return; // not ready yet — idle loop will retry next iteration
	}
	if CONT_SPAWNED.swap(true, Ordering::SeqCst) {
		return;
	}
	spawn_continuation(cont_harness_entry);
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
