.section .text
.extern do_bad_mode
.extern do_irq
.extern do_fiq
.extern do_sync
.extern do_error
.extern D4_TRACE
	.extern get_last_stack_pointer

.macro trap_entry spsel
     stp x29, x30, [sp, #-16]!
     stp x27, x28, [sp, #-16]!
     stp x25, x26, [sp, #-16]!
     stp x23, x24, [sp, #-16]!
     stp x21, x22, [sp, #-16]!
     stp x19, x20, [sp, #-16]!
     stp x17, x18, [sp, #-16]!
     stp x15, x16, [sp, #-16]!
     stp x13, x14, [sp, #-16]!
     stp x11, x12, [sp, #-16]!
     stp x9, x10, [sp, #-16]!
     stp x7, x8, [sp, #-16]!
     stp x5, x6, [sp, #-16]!
     stp x3, x4, [sp, #-16]!
     stp x1, x2, [sp, #-16]!

     mrs x22, tpidr_el0
     stp x22, x0, [sp, #-16]!

     mrs x23, sp_el0
     mrs x22, spsr_el1
     stp x22, x23, [sp, #-16]!

     mrs x23, elr_el1
     mov x22, #\spsel
     stp x22, x23, [sp, #-16]!
.endm

.macro trap_exit
     ldp x22, x23, [sp], #16
     msr elr_el1, x23

     ldp x22, x23, [sp], #16
     msr spsr_el1, x22
     msr sp_el0, x23

     ldp x22, x0, [sp], #16
     msr tpidr_el0, x22

     ldp x1, x2, [sp], #16
     ldp x3, x4, [sp], #16
     ldp x5, x6, [sp], #16
     ldp x7, x8, [sp], #16
     ldp x9, x10, [sp], #16
     ldp x11, x12, [sp], #16
     ldp x13, x14, [sp], #16
     ldp x15, x16, [sp], #16
     ldp x17, x18, [sp], #16
     ldp x19, x20, [sp], #16
     ldp x21, x22, [sp], #16
     ldp x23, x24, [sp], #16
     ldp x25, x26, [sp], #16
     ldp x27, x28, [sp], #16
     ldp x29, x30, [sp], #16
     // D4: for EL1t return (SPSR SPSEL==0), set SP_EL1 = the CURRENT task's
     // scratch slot (per-task exception slot design). The slot TOP was
     // published by the switch path into CoreLocal.scratch_slot (offset 24).
     // This makes SP_EL1 per-task instead of the shared per-core E.
     // For EL1h return (SPSEL==1), SP_EL1 MUST stay = the incoming kernel-stack
     // frame (already set by `mov sp, x0`); overwriting it would strip EL1h
     // tasks of their stack.
     // CoreLocal is #[repr(C)]; scratch_slot is at offset 24 (after
     // exception_sp@0 / this@8 / kernel_sp@16 — see design D-1).
     sub  x22, sp, #288            // x22 = F (frame base; task_x22 preserved in frame)
     str  x21, [x22, #0xd0]        // preserve task_x21 (x21 about to be scratch)
     str  x23, [x22, #0xe0]        // preserve task_x23 (D4-TRACE scratch)
     str  x24, [x22, #0xe8]        // preserve task_x24 (D4-TRACE scratch)
     // Load scratch_slot + saved SPSR for the SP_EL1 gate (below).
     mrs  x21, tpidr_el1           // x21 = &CoreLocal (TPIDR_EL1)
     ldr  x21, [x21, #24]          // x21 = scratch_slot (offset 24) = current task's slot TOP (or kstack top)
     str  x20, [x22, #0xc8]        // preserve task_x20 FIRST (frame slot @ 40+8*20 = 0xc8)
     mrs  x20, spsr_el1            // THEN x20 = saved SPSR_EL1 (scratch; restored at tail)
     // SP_EL1 gate (per-task-exception-slot-design.md R4; nested-EL1h fix):
     //   - EL1t return (SPSR.SPSEL==0): the resumed context uses SP_EL0, so
     //     SP_EL1 is "free" -> stage it to scratch_slot so the NEXT exception
     //     builds its frame on the per-task slot (not the shared per-core E).
     //     This is the path idle takes (create_stack_frame sets spsel=0, and
     //     H-a4 preserves idle's crafted frame), so idle is correct here.
     //   - EL1h return (SPSR.SPSEL==1): the resumed context USES SP_EL1 as its
     //     stack. The 18 ldp pairs already restored sp to that live stack;
     //     forcing scratch_slot would destroy a nested-EL1h handler / kernel
     //     context (PROVEN by D4-TRACE clobbered_live_stack? true at fault
     //     PC 0x4115589c). Leave sp untouched.
     tst  x20, #1                  // SPSR.SPSEL (bit 0): 0=EL1t, 1=EL1h
     b.ne 2f                       // EL1h: leave SP_EL1 = restored live stack
     mov  sp, x21                  // EL1t: SP_EL1 = task's scratch slot (staged)
2:
     // === D4-TRACE (temporary): record the SP_EL1 decision on EVERY return. ===
     adrp x23, D4_TRACE
     add  x23, x23, #:lo12:D4_TRACE
     str  x20, [x23, #0]           // [0] = saved SPSR (M[3:0]: 0b0101=EL1h, 0b0100=EL1t)
     mov  x24, sp
     str  x24, [x23, #8]           // [1] = resume SP (post-gate: slot for EL1t, live stack for EL1h)
     str  x21, [x23, #16]          // [2] = scratch_slot
     mrs  x24, elr_el1
     str  x24, [x23, #24]          // [3] = ELR being returned to
     ldr  x24, [x23, #32]
     add  x24, x24, #1
     str  x24, [x23, #32]          // [4] = call counter++
     mov  x24, #0xd4
     str  x24, [x23, #40]          // [5] = magic (trace written)
     ldr  x21, [x22, #0xd0]        // restore task_x21
     ldr  x23, [x22, #0xe0]        // restore task_x23
     ldr  x24, [x22, #0xe8]        // restore task_x24
     ldr  x20, [x22, #0xc8]        // restore task_x20
     ldr  x22, [x22, #0xd8]        // restore task_x22 (via F, MUST be last)
.endm

/*
 * Exception vector entry
 */
.macro ventry label
.align  7
b       \label
.endm

.macro invalid, reason
mov     x0, sp
mov     x1, #\reason
b       do_bad_mode
.endm

/*
 * Phase 4 double-fault detection (option-d-per-task-slot-rebased.md §5 / NEW-4).
 *
 * Runs at vector entry BEFORE trap_entry, so a bad/overflowing stack is caught
 * BEFORE we push 288 bytes onto it. The two vector families need DIFFERENT
 * predicates (R4.1 lesson — vector selection encodes the interrupted stack):
 *
 *   - el1_sp0_* (exception from EL1t; interrupted context on SP_EL0): the D4
 *     tail staged SP_EL1 = the current task's scratch slot, so SP_EL1 MUST
 *     equal CoreLocal.scratch_slot exactly. Mismatch = D4 tail bypassed or
 *     nested during the tail itself -> fatal (df_check_el1t).
 *
 *   - el1_sync/el1_error (exception from EL1h; interrupted context ON SP_EL1):
 *     sp is a live kernel/handler stack; equality with scratch_slot is the
 *     WRONG predicate. The fatal signal is sp inside the CURRENT SLOT's danger
 *     window [slot_bottom - GUARD, slot_bottom + MARGIN): a handler running on
 *     the slot has overflowed (sp in the guard) or is about to (the next
 *     handler chain won't fit) (df_check_el1h). Kernel stacks live in other
 *     sections and can never land in this window.
 *
 * Scratch discipline: every GPR is live pre-trap_entry. x0 is stashed in
 * TPIDRRO_EL0 (unused by kernel & newlib TLS — they use tpidr_el0/tpidr_el1).
 * The check touches memory exactly once (ldr from CoreLocal, always mapped),
 * so the prologue itself cannot fault. DAIF is masked at entry.
 *
 * Immediates are cross-checked at compile time in interrupts.rs (DF_* consts):
 *   0x11000 = EXCEPTION_SLOT_SIZE + EXCEPTION_SLOT_GUARD (slot-top -> guard base)
 *   0x4000  = EXCEPTION_SLOT_GUARD + DF_SLOT_MARGIN danger window
 *   0x9000  = KERNEL_STACK_SIZE + guard  (.overflow_stacks per-core stride)
 *   0x8000  = KERNEL_STACK_SIZE          (.overflow_stacks top offset)
 *
 * Fatal landing (df_fatal): switch sp to this core's .overflow_stacks tier
 * (MPIDR Aff0-indexed — a runaway task cannot consume its own rescue stack),
 * rebuild a normal trap frame there, and call do_double_fault (-> !).
 * ORIGINAL x1 IS SACRIFICED in the fatal path to carry the bad SP_EL1 into
 * the frame/handler (x2's original value IS preserved in the frame; the
 * diagnosis lives in ESR/FAR/ELR + the bad SP, not in x1).
 */
.macro df_fatal spsel, flavor
	// __start_overflow_stacks is a SINGLE (one-core) rescue block in link.x
	// (== KERNEL_STACK_SIZE + GUARD = 0x9000). Land at its top. No core
	// indexing: this is a halt path, so there is no "own stack" to avoid
	// (and indexing would walk past the 1-core section on core>=1).
	adrp x0, __start_overflow_stacks
	// INVARIANT (review A2): `adrp`/`add` addressing is limited to a 4 GiB
	// page-aligned range relative to the PC. This is always satisfied because
	// the kernel image (text + .overflow_stacks) is linked as a single,
	// contiguous ~tens-of-MiB block, so any section symbol is well within 4 GiB
	// of any handler. If link.x ever moves .overflow_stacks (or any referenced
	// section) gigabytes away from the exception vectors, this `adrp` silently
	// truncates and the rescue path lands on the wrong stack. The const _: ()
	// asserts in interrupts.rs pin the SIZE/STRIDE immediates, but NOT the
	// addressability; a linker-script ASSERT on the section's distance from
	// _start is the durable guard if the layout ever changes.
	add  x0, x0, #:lo12:__start_overflow_stacks
	add  x0, x0, #0x8, lsl #12       // + KERNEL_STACK_SIZE = rescue top
	mov  x1, sp                      // x1 = the BAD SP_EL1 (for diagnostics)
	mov  sp, x0                      // land on the rescue stack
	mrs  x0, tpidrro_el0             // restore the task's original x0
	trap_entry \spsel                // frame on rescue stack (x1 slot = bad sp)
	mov  x0, sp
	// a0 = state (frame). The bad SP_EL1 was stored into the frame's x1 slot
	// by trap_entry; reload it as a1 (x1). flavor is a2 (x2).
	// Frame layout (trap_entry): +0 spsel, +8 elr, +16 spsr, +24 sp_el0,
	// +32 tpidr, +40 x0, +48 x1, +56 x2, ... — x1 slot is +0x30.
	// (R5.1: was 0x8, which is the ELR slot — harness caught the mislabel.)
	ldr  x1, [x0, #0x30]             // a1 = bad SP_EL1 (frame x1 slot)
	mov  x2, #\flavor                // a2 = flavor
	bl   do_double_fault             // -> ! (never returns)
.endm

.macro df_check_el1t flavor
	msr  tpidrro_el0, x0
	mrs  x0, tpidr_el1               // &CoreLocal
	ldr  x0, [x0, #24]               // scratch_slot (slot top; offset 24 = E1)
	cmp  sp, x0                      // EL1t source: SP_EL1 MUST == slot top
	b.ne 9f
	mrs  x0, tpidrro_el0             // ok: restore and fall through
	b    8f
9:	df_fatal 0, \flavor
8:
.endm

.macro df_check_el1h flavor
	msr  tpidrro_el0, x0
	mrs  x0, tpidr_el1               // &CoreLocal
	// R3.1: a sync fault taken from INSIDE a nested RT handler has
	// rt_nest_depth (offset 32) > 0 and SP_EL1 on the exception stack E (not a
	// task's scratch_slot). Validate against the E window; the original
	// scratch_slot check would FALSELY trip here. Off-core / non-nested (==0)
	// stays on the task slot and uses the original check.
	ldr  x1, [x0, #32]               // rt_nest_depth
	cbz  x1, df_slotcheck_\flavor    // ==0 -> task context (scratch_slot)
	// nested RT handler: E window = [e_top - DEFAULT_STACK_SIZE, e_top]
	ldr  x0, [x0, #0]                // exception_sp (e_top)
	sub  x0, x0, #0x2, lsl #12       // e_top - 0x20000 (DEFAULT_STACK_SIZE E)
	sub  x0, sp, x0                  // delta = sp - e_base
	cmp  x0, #0x2, lsl #12           // delta < 0x20000?
	b.lo df_ok_\flavor               // ok: sp inside E
	mrs  x0, tpidrro_el0
	b    df_fail_\flavor
df_slotcheck_\flavor:
	ldr  x0, [x0, #24]               // scratch_slot (slot top)
	sub  x0, x0, #0x11, lsl #12      // slot_bottom - GUARD (top - 0x11000)
	sub  x0, sp, x0                  // delta = sp - guard_base (unsigned)
	cmp  x0, #0x4, lsl #12           // delta < 0x4000 (GUARD 0x1000 + MARGIN 0x3000)?
	b.lo df_ok_\flavor               // -> sp in guard or about to blow the slot
	mrs  x0, tpidrro_el0             // ok: restore and fall through
	b    df_fail_\flavor
df_ok_\flavor:
	mrs  x0, tpidrro_el0             // restore saved x0
	b    df_end_\flavor
df_fail_\flavor:
	df_fatal 1, \flavor
df_end_\flavor:
.endm


/*
 * SYNC exception handler.
 */
.align 6
el1_sync:
	// Phase 5 injection harness (feature "double-fault-injection"): redirect
	// SP_EL1 to INSIDE the current slot's danger window so the REAL
	// df_check_el1h predicate below trips and routes to do_double_fault. This
	// exercises the genuine NEW-4 path (not a synthetic bl), proving the EL1h
	// danger-window detection actually fires on a real exception. The redirect
	// lands SP_EL1 at slot_top - 0xF000, inside df_check_el1h's window
	// [slot_top - 0x11000, slot_top - 0xD000) (delta = 0x2000, caught by the
	// `delta < 0x4000` check). It must run BEFORE df_check_el1h, which reads
	// SP_EL1. This block is compiled ONLY with the `double-fault-injection`
	// feature (DF_INJECT_ON is `.set` in the global_asm! concat in mod.rs).
	//
	// ONE-SHOT BY EXCEPTION IDENTITY: only the injected `udf #0` (ESR EC==0x00,
	// "unknown instruction") is redirected. Any other exception — including a
	// re-fault from do_double_fault's own logging — is left untouched, so the
	// harness cannot recurse. (We cannot use a cross-language Rust flag here:
	// the asm `adrp`/GOT access to a `static` read 0 at EL1; gating on the
	// uniquely-identifiable udf ESR is the robust, side-effect-free alternative.)
	.if DF_INJECT_ON
	mrs  x9, esr_el1
	ubfx x9, x9, #26, #6            // extract ESR EC[31:26]
	// 0x00 = unknown instruction (our injected `udf #0`). Any other EC
	// (e.g. the re-fault from logging) is NOT redirected.
	cmp  x9, #0x00
	b.ne 1f
	mrs  x11, tpidr_el1           // &CoreLocal
	ldr  x11, [x11, #24]          // scratch_slot (slot top)
	sub  x11, x11, #0xF, lsl #12  // slot_top - 0xF000 (inside danger window)
	mov  sp, x11                  // SP_EL1 now in the slot danger window
1:
	.endif
	// Phase 4: double-fault check FIRST (before trap_entry pushes 288 B onto
	// a possibly-bad stack). EL1h source: danger-window predicate.
	df_check_el1h 0
	// No pool-select block: trap_entry builds the 288-byte frame on the
	// task's scratch_slot (staged by the D4 tail), identical to el1_irq/fiq.
	// Capture entry SP_EL1 for the NEW-1 assert WITHOUT clobbering any task
	// register: after trap_entry, sp = frame_base = (entry SP_EL1) - 288, so
	// `add x1, sp, #288` reconstructs the entry SP_EL1. Do NOT `mov x1, sp`
	// BEFORE trap_entry — trap_entry's first insn (`stp x1,x2`) would then save
	// a corrupted task x1 into the frame. `mrs sp_el1` is UNDEFINED at EL1.
	trap_entry 1
	mov	x0, sp
	add	x1, sp, #288
	bl	do_sync
	trap_exit
	eret
	// speculation barrier after the ERET to prevent the CPU
	// from speculating past the exception return.
	dsb	nsh
	isb
.size el1_sync, .-el1_sync
.type el1_sync, @function

/*
 * IRQ handler.
 *
 * The trap_entry frame lives on the task's OWN scratch slot (SP_EL1 ==
 * CoreLocal.scratch_slot, staged by the D4 tail on the preceding exception
 * return). The slot is per-task, so it survives across context switches with no
 * D3 copy. (Pre-1b′ comment about "the task's kernel stack" is obsolete: the
 * frame is on the per-task exception slot, not the kernel stack.)
 */
.align 6
el1_irq:
	trap_entry 1
      mov     x0, sp
      bl      do_irq
      cmp x0, 0
      b.eq 1f
      // switch to the next task
      // Per-task exception slot design (per-task-exception-slot-design.md):
      // The old task's trap frame already lives on ITS OWN scratch slot
      // (SP_EL1 == scratch_slot at entry). No D3 copy. Record the old task's
      // frame base, then load the new task's slot.
      mov x1, sp                       /* x1 = old frame base (on old task's slot) */
      str x1, [x0]                     /* *x0 = old task's frame base (unchanged) */
      bl get_last_stack_pointer     /* get new sp (frame base in new task's slot) */
      mov sp, x0                       /* SP_EL1 = new task's slot (via D4 tail) */
      // Publish new task's scratch-slot TOP for the D4 tail to load.
      // scratch_slot is @24 (NOT kernel_sp @16, which stays the 128KiB
      // kernel-stack top for call_with_kernel_stack — see D-1 in the doc).
      add x1, x0, #288                /* x1 = new task's scratch slot TOP (= frame base + 288) */
      mrs x2, tpidr_el1
      str x1, [x2, #24]             /* CoreLocal.scratch_slot = slot TOP */
1:
      trap_exit
      eret
      // speculation barrier after the ERET to prevent the CPU
      // from speculating past the exception return.
      dsb     nsh
      isb
.size el1_irq, .-el1_irq
.type el1_irq, @function

/*
 * FIQ handler.
 *
 * Same as el1_irq: the trap_entry frame lives on the task's OWN scratch slot
 * (SP_EL1 == CoreLocal.scratch_slot, staged by the D4 tail), not the kernel
 * stack. Per-task slot survives context switches with no D3 copy.
 */
.align 6
el1_fiq:
	trap_entry 1
      mov     x0, sp
      bl      do_fiq
      cmp x0, 0
      b.eq 1f
      // switch to the next task
      // Per-task exception slot design (per-task-exception-slot-design.md):
      // The old task's trap frame already lives on ITS OWN scratch slot
      // (SP_EL1 == scratch_slot at entry). No D3 copy. Record the old task's
      // frame base, then load the new task's slot.
      mov x1, sp                       /* x1 = old frame base (on old task's slot) */
      str x1, [x0]                     /* *x0 = old task's frame base (unchanged) */
      bl get_last_stack_pointer     /* get new sp (frame base in new task's slot) */
      mov sp, x0                       /* SP_EL1 = new task's slot (via D4 tail) */
      // Publish new task's scratch-slot TOP for the D4 tail to load.
      // scratch_slot is @24 (NOT kernel_sp @16, which stays the 128KiB
      // kernel-stack top for call_with_kernel_stack — see D-1 in the doc).
      add x1, x0, #288                /* x1 = new task's scratch slot TOP (= frame base + 288) */
      mrs x2, tpidr_el1
      str x1, [x2, #24]             /* CoreLocal.scratch_slot = slot TOP */
1:
      trap_exit
      eret
      // speculation barrier after the ERET to prevent the CPU
      // from speculating past the exception return.
      dsb     nsh
      isb
.size el1_fiq, .-el1_fiq
.type el1_fiq, @function

.align 6
el1_error:
	// Phase 4: double-fault check FIRST (EL1h source: danger-window predicate).
	df_check_el1h 1
	// No pool-select; trap_entry builds the frame on the task's scratch_slot.
	// Capture entry SP_EL1 as frame_base+288 AFTER trap_entry (do NOT clobber
	// task x1 before trap_entry; `mrs sp_el1` UNDEFINED at EL1). See el1_sync.
	trap_entry 1
      mov     x0, sp
      add     x1, sp, #288
      bl      do_error
      trap_exit
      eret
      // speculation barrier after the ERET to prevent the CPU
      // from speculating past the exception return.
      dsb     nsh
      isb
.size el1_error, .-el1_error
.type el1_error, @function

/*
 * SYNC exception handler with SP0.
 */
.align 6
el1_sp0_sync:
      msr spsel, #1            // select SP_EL1 (staged = task's scratch_slot by D4 tail)
      // Phase 4: EL1t source — SP_EL1 must EQUAL the task's scratch slot.
      df_check_el1t 2
      trap_entry 0
      mov     x0, sp
      add     x1, sp, #288     // NEW-1: entry SP_EL1 = frame_base + 288 (see el1_sync)
      bl      do_sync
      // Restore scratch_slot = entry SP_EL1 (do_sync cleared it), so that
      // trap_exit's D4 tail correctly sets SP_EL1 for the return. Without this,
      // the next exception's df_check_el1t may find SP_EL1=0 != scratch_slot.
      // x1 is clobbered by do_sync (caller-saved), so recompute from frame_base.
      mrs     x2, tpidr_el1
      add     x1, sp, #288
      str     x1, [x2, #24]
      trap_exit
      eret
      // speculation barrier after the ERET to prevent the CPU
      // from speculating past the exception return.
      dsb     nsh
      isb
.size el1_sp0_sync, .-el1_sp0_sync
.type el1_sp0_sync, @function

/*
 * IRQ handler with SP0.
 */
.align 6
el1_sp0_irq:
      msr spsel, #1            // select SP_EL1 (=E, set at boot + trap_exit tail)
      trap_entry 0
      mov     x0, sp
      bl      do_irq
      cmp x0, 0
      b.eq 1f
      // switch to the next task
      // Per-task exception slot design (per-task-exception-slot-design.md):
      // The old task's trap frame already lives on ITS OWN scratch slot
      // (SP_EL1 == scratch_slot at entry). No D3 copy. Record the old task's
      // frame base, then load the new task's slot.
      mov x1, sp                       /* x1 = old frame base (on old task's slot) */
      str x1, [x0]                     /* *x0 = old task's frame base (unchanged) */
      bl get_last_stack_pointer     /* get new sp (frame base in new task's slot) */
      mov sp, x0                       /* SP_EL1 = new task's slot (via D4 tail) */
      // Publish new task's scratch-slot TOP for the D4 tail to load.
      // scratch_slot is @24 (NOT kernel_sp @16, which stays the 128KiB
      // kernel-stack top for call_with_kernel_stack — see D-1 in the doc).
      add x1, x0, #288                /* x1 = new task's scratch slot TOP (= frame base + 288) */
      mrs x2, tpidr_el1
      str x1, [x2, #24]             /* CoreLocal.scratch_slot = slot TOP */
1:
      trap_exit
      eret
      // speculation barrier after the ERET to prevent the CPU
      // from speculating past the exception return.
      dsb     nsh
      isb
.size el1_sp0_irq, .-el1_sp0_irq
.type el1_sp0_irq, @function

/*
 * FIQ handler with SP0.
 */
.align 6
el1_sp0_fiq:
      msr spsel, #1            // select SP_EL1 (=E, set at boot + trap_exit tail)
      trap_entry 0
      mov     x0, sp
      bl      do_fiq
      cmp x0, 0
      b.eq 1f
      // switch to the next task
      // Per-task exception slot design (per-task-exception-slot-design.md):
      // The old task's trap frame already lives on ITS OWN scratch slot
      // (SP_EL1 == scratch_slot at entry). No D3 copy. Record the old task's
      // frame base, then load the new task's slot.
      mov x1, sp                       /* x1 = old frame base (on old task's slot) */
      str x1, [x0]                     /* *x0 = old task's frame base (unchanged) */
      bl get_last_stack_pointer     /* get new sp (frame base in new task's slot) */
      mov sp, x0                       /* SP_EL1 = new task's slot (via D4 tail) */
      // Publish new task's scratch-slot TOP for the D4 tail to load.
      // scratch_slot is @24 (NOT kernel_sp @16, which stays the 128KiB
      // kernel-stack top for call_with_kernel_stack — see D-1 in the doc).
      add x1, x0, #288                /* x1 = new task's scratch slot TOP (= frame base + 288) */
      mrs x2, tpidr_el1
      str x1, [x2, #24]             /* CoreLocal.scratch_slot = slot TOP */
1:
      trap_exit
      eret
      // speculation barrier after the ERET to prevent the CPU
      // from speculating past the exception return.
      dsb     nsh
      isb
.size el1_sp0_fiq, .-el1_sp0_fiq
.type el1_sp0_fiq, @function

.align 6
el1_sp0_error:
      msr spsel, #1            // select SP_EL1 (staged = task's scratch_slot by D4 tail)
      // Phase 4: EL1t source — SP_EL1 must EQUAL the task's scratch slot.
      df_check_el1t 3
      trap_entry 0
      mov     x0, sp
      add     x1, sp, #288     // NEW-1: entry SP_EL1 = frame_base + 288 (see el1_sync)
      bl      do_error
      trap_exit
      eret
      // speculation barrier after the ERET to prevent the CPU
      // from speculating past the exception return.
      dsb     nsh
      isb
.size el1_sp0_error, .-el1_sp0_error
.type el1_sp0_error, @function

el0_sync_invalid:
   invalid 0
.type el0_sync_invalid, @function

el0_irq_invalid:
   invalid 1
.type el0_irq_invalid, @function

el0_fiq_invalid:
   invalid 2
.type el0_fiq_invalid, @function

el0_error_invalid:
   invalid 3
.type el0_error_invalid, @function

el1_sync_invalid:
   invalid 0
.type el1_sync_invalid, @function

el1_irq_invalid:
   invalid 1
.type el1_irq_invalid, @function

el1_fiq_invalid:
   invalid 2
.type el1_fiq_invalid, @function

el1_error_invalid:
   invalid 3
.type el1_error_invalid, @function

	/* Exception vectors.
	 *
	 * Must reside in executable, mapped memory very early in boot, since `_start`
	 * programs `VBAR_EL1` before the full kernel paging setup.
	 */
	.section .vectors, "ax"
		.align  11
		.global vector_table
	.type vector_table, @function
	vector_table:
/* Current EL with SP0 */
ventry el1_sp0_sync             // Synchronous EL1t
ventry el1_sp0_irq              // IRQ EL1t
ventry el1_sp0_fiq              // FIQ EL1t
ventry el1_sp0_error            // Error EL1t

/* Current EL with SPx */
ventry el1_sync                 // Synchronous EL1h
ventry el1_irq                  // IRQ EL1h
ventry el1_fiq                  // FIQ EL1h
ventry el1_error                // Error EL1h

/* Lower EL using AArch64 */
ventry el0_sync_invalid         // Synchronous 64-bit EL0
ventry el0_irq_invalid          // IRQ 64-bit EL0
ventry el0_fiq_invalid          // FIQ 64-bit EL0
ventry el0_error_invalid        // Error 64-bit EL0

/* Lower EL using AArch32 */
ventry el0_sync_invalid         // Synchronous 32-bit EL0
ventry el0_irq_invalid          // IRQ 32-bit EL0
ventry el0_fiq_invalid          // FIQ 32-bit EL0
ventry el0_error_invalid        // Error 32-bit EL0
	.size vector_table, .-vector_table
	.section .rodata
	// Keep a second global alias that is easy to find in debuggers/symbolizers.
	.global __hermit_vector_table
	.type __hermit_vector_table, @function
__hermit_vector_table = vector_table
	// NOTE: `__hermit_vector_table` is an absolute symbol alias, so we must not
	// emit a `.size` directive for it (LLVM requires size expressions be absolute).

	// Export a stable, prefixed symbol name. The build may also apply a global
	// prefix automatically, but this guarantees `hermit_vector_table` exists.
	.global hermit_vector_table
	.type hermit_vector_table, @function
hermit_vector_table = vector_table
	// NOTE: `hermit_vector_table` is an absolute symbol alias; no `.size`.
