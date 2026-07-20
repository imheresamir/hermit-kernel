.section .text
.extern do_bad_mode
.extern do_irq
.extern do_fiq
.extern do_sync
.extern do_error
	.extern get_last_stack_pointer
	.extern HERMIT_EARLY_EXCEPTION_STACK_POOL

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
     // D4 (trap_exit tail) DEFERRED to 2a.3 (§8.13). The tail below sets
     // SP_EL1 = E on EL1t return to maintain INV-D. It is CORRECT in isolation
     // (x22 holds F throughout; x21/x22 reloaded from frame slots F+0xd0/F+0xd8),
     // but it must NOT go live until create_stack_frame gives tasks a valid EL1t
     // kernel-stack model. Under the current (pre-2a.3) model tasks run EL1t with
     // SP_EL1 = task KERNEL stack; firing this tail makes the task re-enter
     // exceptions on E, its next frame lands on the shared E, last_stack_pointer
     // goes stale, and a later switch-back pops garbage -> "Unsupported exception
     // class: 0x0" (PC=0x410afd90, TaskId(0)). See §10.10.2. Re-enable together
     // with the create_stack_frame EL1t conversion.
     //
     // sub  x22, sp, #288            // x22 = F (frame base)
     // str  x21, [x22, #0xd0]        // task_x21 -> frame slot (x21 about to be scratch)
     // mrs  x21, spsr_el1
     // tst  x21, #1                  // SPSEL bit (0 = EL1t, 1 = EL1h)
     // b.ne 1f                       // EL1h: keep SP_EL1 = incoming kernel-stack frame
     // mrs  x21, tpidr_el1           // x21 = &CoreLocal (TPIDR_EL1)
     // ldr  x21, [x21, #0]           // x21 = E (exception_sp, offset 0)
     // mov  sp, x21                  // SP_EL1 = E
     // 1:
     // ldr  x21, [x22, #0xd0]        // task_x21 (via F in x22)
     // ldr  x22, [x22, #0xd8]        // task_x22 (via F in x22)
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
 * SYNC exception handler.
 */
.align 6
el1_sync:

		// Switch to an early emergency exception stack from a bounded pool.
		//
		// slot = (MPIDR_EL1 & 0xff) & (POOL_SIZE-1)
		// sp = &HERMIT_EARLY_EXCEPTION_STACK_POOL + slot*64KiB + 64KiB
		mrs     x24, mpidr_el1
		and     x24, x24, #0xff
		and     x24, x24, #0x3f
		adrp    x25, HERMIT_EARLY_EXCEPTION_STACK_POOL
		add     x25, x25, #:lo12:HERMIT_EARLY_EXCEPTION_STACK_POOL
		mov     x26, #65536
		mul     x24, x24, x26
		add     x24, x25, x24
		add     x24, x24, x26
		mov     sp, x24
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
 * IRQ handler.
 *
 * IMPORTANT: We do NOT switch to the emergency stack here. The trap_entry
 * frame must live on the task's own kernel stack so that it survives across
 * context switches. If we used the emergency stack, a subsequent exception
 * on the same core would reset SP to the top of the emergency stack and the
 * new trap_entry would overwrite the previously saved State, corrupting the
 * task's registers (x30/LR, elr_el1, etc.). This was the root cause of
 * crashes with PC=0x0 / x30=0x0 after the executor parks.
 */
.align 6
el1_irq:
	trap_entry 1
      mov     x0, sp
      bl      do_irq
      cmp x0, 0
      b.eq 1f
      // switch to the next task
      mov x1, sp
      str x1, [x0]                  /* store old sp */
      bl get_last_stack_pointer     /* get new sp   */
      mov sp, x0
      // D1: record the incoming task's restored SP_EL1 for call_with_kernel_stack (D5).
      // trap_exit pops exactly STATE_SIZE (288) bytes from the saved frame.
      add x1, x0, #288
      mrs x2, tpidr_el1
      str x1, [x2, #16]             /* CoreLocal.kernel_sp = kernel stack top */
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
 * Same rationale as el1_irq: use the task's kernel stack directly to
 * avoid emergency stack corruption across context switches.
 */
.align 6
el1_fiq:
	trap_entry 1
      mov     x0, sp
      bl      do_fiq
      cmp x0, 0
      b.eq 2f
      // switch to the next task
      mov x1, sp
      str x1, [x0]                  /* store old sp */
      bl get_last_stack_pointer     /* get new sp   */
      mov sp, x0
      // D1: record the incoming task's restored SP_EL1 for call_with_kernel_stack (D5).
      // trap_exit pops exactly STATE_SIZE (288) bytes from the saved frame.
      add x1, x0, #288
      mrs x2, tpidr_el1
      str x1, [x2, #16]             /* CoreLocal.kernel_sp = kernel stack top */
2:
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
	mrs     x24, mpidr_el1
	and     x24, x24, #0xff
	and     x24, x24, #0x3f
	adrp    x25, HERMIT_EARLY_EXCEPTION_STACK_POOL
	add     x25, x25, #:lo12:HERMIT_EARLY_EXCEPTION_STACK_POOL
	mov     x26, #65536
	mul     x24, x24, x26
	add     x24, x25, x24
	add     x24, x24, x26
	mov     sp, x24
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
      msr spsel, #1            // select SP_EL1 (=E, set at boot + trap_exit tail)
      trap_entry 0
      mov     x0, sp
      bl      do_sync
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
      b.eq 3f
      // switch to the next task
      mov x1, sp
      str x1, [x0]                  /* store old sp */
      bl get_last_stack_pointer     /* get new sp   */
      mov sp, x0
      // D1: record the incoming task's restored SP_EL1 for call_with_kernel_stack (D5).
      // trap_exit pops exactly STATE_SIZE (288) bytes from the saved frame.
      add x1, x0, #288
      mrs x2, tpidr_el1
      str x1, [x2, #16]             /* CoreLocal.kernel_sp = kernel stack top */
3:
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
      b.eq 4f
      // switch to the next task
      mov x1, sp
      str x1, [x0]                  /* store old sp */
      bl get_last_stack_pointer     /* get new sp   */
      mov sp, x0
      // D1: record the incoming task's restored SP_EL1 for call_with_kernel_stack (D5).
      // trap_exit pops exactly STATE_SIZE (288) bytes from the saved frame.
      add x1, x0, #288
      mrs x2, tpidr_el1
      str x1, [x2, #16]             /* CoreLocal.kernel_sp = kernel stack top */
4:
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
      msr spsel, #1            // select SP_EL1 (=E, set at boot + trap_exit tail)
      trap_entry 0
      mov     x0, sp
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
