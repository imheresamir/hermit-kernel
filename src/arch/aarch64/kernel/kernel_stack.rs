use core::mem;

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
		// Save return address on the CURRENT stack (SP_EL0 = task body stack) before switching.
		"str x30, [sp, #-16]!",
		// Select SP_EL1 so we can write it. Under INV-D SP_EL1 == E (exception
		// stack), NOT the task kernel stack, so we must load the kernel stack
		// explicitly from CoreLocal.kernel_sp (offset 16) instead of relying on
		// SP_EL1 already holding it (that assumption is what the old code made).
		"msr spsel, #1",
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
		// Switch back to user/body stack (SP_EL0), restoring SP_EL1 = E for the
		// next exception (INV-D).
		"msr spsel, #0",
		// Restore return address from the stack
		"ldr x30, [sp], 16",
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
