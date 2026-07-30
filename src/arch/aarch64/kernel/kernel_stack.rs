use core::mem;

use crate::arch::aarch64::kernel::core_local::CoreLocal;

type Reg = mem::MaybeUninit<usize>;

#[unsafe(naked)]
pub unsafe extern "C" fn call_with_kernel_stack(
	x0: Reg,
	x1: Reg,
	x2: Reg,
	x3: Reg,
	x4: Reg,
	x5: Reg,
	f: unsafe extern "C" fn(x0: Reg, x1: Reg, x2: Reg, x3: Reg, x4: Reg, x5: Reg) -> Reg,
) -> Reg {
	core::arch::naked_asm!(
		// Save return address (x30) AND callee-saved x19 on SP_EL0 (task body stack)
		// before switching. x19 will hold the original SP_EL1 across the function
		// call (the callee preserves x19 per the AArch64 C ABI), so we restore the
		// EXACT original SP_EL1 afterward — NOT a re-read of CoreLocal.scratch_slot,
		// which may have been cleared (set to 0) by continuations park/drain logic
		// during the function call.
		"stp x30, x19, [sp, #-16]!",
		// Select SP_EL1 so we can write it. Under INV-D SP_EL1 == E (exception
		// stack), NOT the task kernel stack, so we must load the kernel stack
		// explicitly from CoreLocal.kernel_sp (offset 16) instead of relying on
		// SP_EL1 already holding it (that assumption is what the old code made).
		"msr spsel, #1",
		// Save the ORIGINAL SP_EL1 before clobbering it with kernel_sp.
		// x19 is callee-saved — the function called via x6 will preserve it.
		"mov x19, sp",
		// x9 = &CoreLocal (per-core, via TPIDR_EL1)
		"mrs x9, tpidr_el1",
		// SP_EL1 = current task's kernel-stack top (kernel_sp field, offset 16).
		// `ldr sp, [...]` is illegal; load to a GPR then `mov sp, x9` (legal EL1 write).
		"ldr x9, [x9, #16]",
		"mov sp, x9",
		// Re-enable IRQs and FIQs
		"msr daifclr, #0b11",
		// Call the function pointer (stored in x6)
		"blr x6",
		// Disable IRQs and FIQs before restoring stack
		"msr daifset, #0b11",
		// Restore the ORIGINAL SP_EL1 (saved in x19 before switching to kernel_sp).
		// This is correct regardless of whether CoreLocal.scratch_slot was cleared
		// during the function call (e.g. by continuations park/drain), because we
		// restore the value that was live at entry — not a potentially-zeroed re-read.
		"mov sp, x19",
		// Publish the restored SP_EL1 as CoreLocal.scratch_slot, overwriting any stale
		// value left by an EL1h IRQ/FIQ switch path (which published
		// kernel_stack_frame_base+288 when switching to/from a task that was interrupted
		// inside an EL1h context — that value is a kernel-stack address, not a slot top).
		// Without this, the next EL1t exception's df_check_el1t will compare SP_EL1
		// (= correct scratch slot) against scratch_slot (= stale kernel stack address)
		// and produce a false double-fault.
		// x9 is caller-saved and was clobbered by the blr x6, so reload it.
		"mrs x9, tpidr_el1",
		"str x19, [x9, #24]",
		// Switch back to user/body stack (SP_EL0).
		"msr spsel, #0",
		// Restore return address and callee-saved x19 from SP_EL0
		"ldp x30, x19, [sp], 16",
		// Re-enable IRQs and FIQs
		"msr daifclr, #0b11",
		// Return to caller (return value is in x0)
		"ret",
	)
}

macro_rules! kernel_function_impl {
	($kernel_function:ident($($arg:ident: $A:ident),*; $($z:ident: Reg),*)) => {
		/// Executes `f` on the kernel stack.
		#[allow(dead_code)]
		#[inline]
		pub unsafe extern "C" fn $kernel_function<R, $($A),*>($($arg: $A,)* f: unsafe extern "C" fn($($A),*) -> R) -> R {
			unsafe {
				$(
					assert!(size_of::<$A>() <= size_of::<Reg>());
				)*
				assert!(size_of::<R>() <= size_of::<Reg>());

				let call_with_kernel_stack = mem::transmute::<*const (), unsafe extern "C" fn(
						$($arg: $A,)*
						$($z: Reg,)*
						f: unsafe extern "C" fn(
							$($arg: $A,)*
						) -> R,
					) -> R>(call_with_kernel_stack as *const ());

				// §4D: verify CoreLocal.kernel_sp was updated by the scheduler before
				// the asm switch (start.s only publishes scratch_slot @24, not kernel_sp
				// @16). A zero or stale value means deep handler work runs on the wrong
				// stack → overflow → silent memory corruption (not caught by hardware
				// until adjacent memory is already clobbered).
				assert!(
					CoreLocal::get().get_kernel_sp() != 0,
					"call_with_kernel_stack: kernel_sp is zero — scheduler did not update CoreLocal.kernel_sp (§4D)"
				);

				$(
					let $z = Reg::uninit();
				)*

				call_with_kernel_stack(
					$($arg,)*
					$($z,)*
					f,
				)
			}
		}
	};
}

kernel_function_impl!(kernel_function0(; u1: Reg, u2: Reg, u3: Reg, u4: Reg, u5: Reg, u6: Reg));
kernel_function_impl!(kernel_function1(arg1: A1; u2: Reg, u3: Reg, u4: Reg, u5: Reg, u6: Reg));
kernel_function_impl!(kernel_function2(arg1: A1, arg2: A2; u3: Reg, u4: Reg, u5: Reg, u6: Reg));
kernel_function_impl!(kernel_function3(arg1: A1, arg2: A2, arg3: A3; u4: Reg, u5: Reg, u6: Reg));
kernel_function_impl!(kernel_function4(arg1: A1, arg2: A2, arg3: A3, arg4: A4; u5: Reg, u6: Reg));
kernel_function_impl!(kernel_function5(arg1: A1, arg2: A2, arg3: A3, arg4: A4, arg5: A5; u6: Reg));
kernel_function_impl!(kernel_function6(arg1: A1, arg2: A2, arg3: A3, arg4: A4, arg5: A5, arg6: A6; ));
