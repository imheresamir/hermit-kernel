.section .text
.extern do_bad_mode
.extern do_irq
.extern do_fiq
.extern do_sync
.extern do_error
	.extern get_last_stack_pointer
	.extern EXCEPTION_STACKS
	.extern IRQ_STACKS
	// {exception_stack_size}, {irq_stack_size}, {kernel_config_offset} are passed
	// by the global_asm! in mod.rs; referenced here (kept as bindings) for any
	// future early-boot core_count / stack-size bounds-checks against
	// KERNEL_CONFIG.

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
 * Symmetric IRQ/FIQ handler body (Option D / Option-A symmetry).
 *
 * Under Option D every exception entry flips to EL1h with SP_EL1 = the
 * per-core EXCEPTION_STACKS[core_id] (set at boot), so SP_EL1 is ALWAYS
 * the exception/IRQ stack (hardware-guaranteed, never switched). We keep
 * it that way and build each task's State frame on the task's OWN SP_EL0
 * stack instead, so the saved context is per-task and survives a context
 * switch (it must NOT live on the shared exception stack, or the next IRQ
 * overwrites it).
 *
 * Sequence:
 *   1. (already EL1h) capture exception-stack top in x22 (handler body
 *      scratch; isolated + single-occupancy per core, protected by IRQ
 *      masking — NOT reentrant: see the warning below).
 *   2. switch working sp to the interrupted task's SP_EL0; trap_entry builds
 *      the State frame there; x21 = frame pointer (persists on the task
 *      stack across the C call).
 *   3. switch sp to the per-core IRQ stack; call the Rust handler with
 *      x0 = old task frame pointer (the do_irq/str/get_last/mov-sp/trap_exit
 *      contract: do_irq returns &current_task.last_stack_pointer; we str the
 *      old frame ptr there, get_last returns the NEW task's frame ptr).
 *      The IRQ stack (32 KB per core, in .irq_stacks) is used instead of the
 *      4 KB exception stack because do_irq -> executor::run() -> deep async
 *      Rust needs far more stack space than 4 KB.
 *   4. switch sp to the new task frame (on its SP_EL0) on context switch,
 *      or restore the old task frame on no-switch.
 *   5. CRITICAL (Option D): restore SP_EL1 = exception-stack top (x22)
 *      BEFORE eret. Under Option D tasks resume at EL1t, so SP_EL1 must
 *      remain the per-core exception stack (NOT the IRQ stack used for the
 *      body); if we leave SP_EL1 = the task frame pointer or IRQ stack,
 *      the next IRQ re-enters with a poisoned SP_EL1 and faults. trap_exit
 *      pops the frame from sp, then we overwrite sp (= SP_EL1) with the
 *      exc top, so eret returns the task to EL1t with SP_EL1 clean.
 *
 * Clobbers x20 (task stack top), x21 (old frame ptr), x22 (IRQ/exc top).
 * x20/x21/x22 are callee-saved (x19-x28) and are preserved across the
 * bl to the Rust handler, and are saved/restored by trap_entry/trap_exit.
 *
 * Note: x25/x26/x28 are also clobbered by load_exc_stack_top and
 * load_irq_stack_top, but they are callee-saved and must be saved by
 * the Rust handler if it uses them (the bl to do_irq/do_fiq handles this).
 */
/*
 * Compute x22 = EXCEPTION_STACKS[core_id] + exception_stack_size, the
 * known-good per-core exception-stack top (set at boot). Used to (re)load
 * SP_EL1 to a valid value after trap_exit, because trap_exit restores x22
 * from the (interrupted/new task's) frame, which does NOT hold the exc-top.
 * Clobbers x25, x26, x28.
 */
.macro load_exc_stack_top
    mrs x25, mpidr_el1
    and x25, x25, #0xff
    adrp x26, EXCEPTION_STACKS
    add x26, x26, #:lo12:EXCEPTION_STACKS
    mov x28, #{exception_stack_size}
    mul x28, x25, x28
    add x22, x26, x28
    add x22, x22, #{exception_stack_size}    // top of stack (grows downward)
.endm

/*
 * Compute x22 = IRQ_STACKS[core_id] + irq_stack_size, the per-core IRQ
 * stack top. Used by irq_handler for the handler body, which needs far
 * more than the 4 KB exception stack (do_irq -> executor::run() ->
 * handle_waiting_tasks() can use 10-20 KB of frames).
 */
.macro load_irq_stack_top
    mrs x25, mpidr_el1
    and x25, x25, #0xff
    adrp x26, IRQ_STACKS
    add x26, x26, #:lo12:IRQ_STACKS
    mov x28, #{irq_stack_size}
    mul x28, x25, x28
    add x22, x26, x28
    add x22, x22, #{irq_stack_size}          // top of stack (grows downward)
.endm

.macro irq_handler target
    msr spsel, #1                 // ensure EL1h; sp == SP_EL1 at entry
    load_exc_stack_top            // x22 = KNOWN-GOOD exception-stack top
                                  // (computed, not captured, so a corrupt
                                  // entry SP_EL1 cannot poison it).
    mrs x20, sp_el0               // x20 = interrupted task's own stack top
    mov sp, x20                   // working sp = task stack
    trap_entry 1                  // State frame on TASK stack; sp = frame ptr
    mov x21, sp                   // x21 = old task frame ptr (persists)
    load_irq_stack_top            // x22 = IRQ stack top for handler body.
                                  // The 4 KB exception stack is too small for
                                  // do_irq -> executor::run() -> deep async Rust.
                                  // IRQ_STACKS (64 KB per core) is purpose-built
                                  // for this: isolated + single-occupancy per
                                  // core. NOT reentrant — the entire handler
                                  // runs with IRQs masked (PSTATE.I set by the
                                  // exception into EL1, never cleared until
                                  // eret), so a second IRQ cannot preempt this
                                  // body. Do NOT enable IRQs inside do_irq: a
                                  // nested IRQ would rebuild its State frame on
                                  // the same task SP_EL0 and reuse this same IRQ
                                  // stack, clobbering the in-flight handler.
    mov sp, x22                   // body on IRQ stack (single-occupancy per core)
    mov x0, x21                  // x0 = old task frame ptr (arg to handler)
    bl  \target
    cmp x0, 0
    b.eq 1f
    str x21, [x0]                // store old frame ptr into scheduler slot
    bl get_last_stack_pointer      // x0 = new task frame ptr (on its SP_EL0)
    mov sp, x0                   // sp = new task frame
    b 3f
1:
    mov sp, x21                   // restore old task frame ptr
3:
    trap_exit                     // pops task frame; restores sp_el0 + spsr (EL1t)
    load_exc_stack_top            // x22 = known-good exc top (frame's x22 slot
                                  // is the task's, not the exc top)
    mov sp, x22                   // CRITICAL: SP_EL1 = exception-stack top (clean for next IRQ)
    eret
    dsb nsh
    isb
.endm

/*
 * SYNC exception handler.
 *
 * Exception entry flips to EL1h with SP_EL1 = the per-core
 * EXCEPTION_STACKS[core_id] (set at boot), so SP_EL1 is ALWAYS the
 * exception stack (hardware-guaranteed). The State frame is built there for
 * the (transient, non-switching) sync/error path; trap_exit restores
 * sp_el0 + spsr and eret returns the task to EL1t. SP_EL1 is never
 * moved here, so it stays the exception stack across the round trip.
 */
.align 6
el1_sync:
	msr spsel, #1                 // EL1h; sp == SP_EL1
	load_exc_stack_top            // x22 = KNOWN-GOOD exception-stack top
	                               // (computed, not captured — self-heals a
	                               // poisoned SP_EL1 left by a prior exception;
	                               // without this, trap_entry's downward frame
	                               // and the do_sync -> abort -> exit ->
	                               // reschedule call chain spill into the
	                               // unmapped guard and fault at [sp,#0x50]).
	mov sp, x22                   // SP_EL1 = exception-stack top
	trap_entry 1
      mov     x0, sp
      bl      do_sync
      trap_exit
      eret
      // speculation barrier after the ERET to prevent the CPU
      // from speculating past the exception return.
      dsb     nsh
      isb
.size el1_sync, .-el1_sync
.type el1_sync, @function

/*
 * IRQ handler — see the `irq_handler` macro: the State frame is built on
 * the interrupted task's OWN SP_EL0 (per-task, survives a switch), while
 * the handler body runs on the per-core IRQ stack (32 KB, isolated and
 * reentrant). SP_EL1 is always restored to the exception stack (4 KB)
 * before eret (Option D requirement).
 */
.align 6
el1_irq:
	irq_handler do_irq
.size el1_irq, .-el1_irq
.type el1_irq, @function

/*
 * FIQ handler — same contract as the IRQ handler above.
 */
.align 6
el1_fiq:
	irq_handler do_fiq
.size el1_fiq, .-el1_fiq
.type el1_fiq, @function

.align 6
el1_error:
	msr spsel, #1                 // EL1h; sp == SP_EL1
	load_exc_stack_top            // x22 = KNOWN-GOOD exception-stack top (self-heal)
	mov sp, x22                   // SP_EL1 = exception-stack top
	trap_entry 1
      mov     x0, sp
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
	msr spsel, #1                 // EL1h; sp == SP_EL1
	load_exc_stack_top            // x22 = KNOWN-GOOD exception-stack top (self-heal)
	mov sp, x22                   // SP_EL1 = exception-stack top
	trap_entry 1
      mov     x0, sp
      bl      do_sync
      trap_exit
      eret
      dsb     nsh
      isb
.size el1_sp0_sync, .-el1_sp0_sync
.type el1_sp0_sync, @function

/*
 * IRQ handler with SP0 — same symmetric contract as el1_irq.
 */
.align 6
el1_sp0_irq:
	irq_handler do_irq
.size el1_sp0_irq, .-el1_sp0_irq
.type el1_sp0_irq, @function

/*
 * FIQ handler with SP0 — same symmetric contract as el1_fiq.
 */
.align 6
el1_sp0_fiq:
	irq_handler do_fiq
.size el1_sp0_fiq, .-el1_sp0_fiq
.type el1_sp0_fiq, @function

.align 6
el1_sp0_error:
	msr spsel, #1                 // EL1h; sp == SP_EL1 (per-core exception stack)
	trap_entry 1
      mov     x0, sp
      bl      do_error
      trap_exit
      eret
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
