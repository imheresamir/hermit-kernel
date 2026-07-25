//! Architecture dependent interface to initialize a task

use core::arch::naked_asm;
use core::sync::atomic::Ordering;

use aarch64_cpu::asm::barrier::{SY, isb};
use aarch64_cpu::registers::*;
use align_address::Align;
use free_list::{PageLayout, PageRange};
use memory_addresses::{PhysAddr, VirtAddr};

use crate::arch::aarch64::kernel::CURRENT_STACK_ADDRESS;
use crate::arch::aarch64::kernel::core_local::core_scheduler;
use crate::arch::aarch64::mm::paging::{BasePageSize, PageSize, PageTableEntryFlags};
use crate::config::{DEFAULT_STACK_SIZE, KERNEL_STACK_SIZE};
use crate::mm::{FrameAlloc, PageAlloc, PageRangeAllocator};
use crate::scheduler::PerCoreSchedulerExt;
use crate::scheduler::task::{Task, TaskFrame, TaskId};

#[derive(Debug)]
// NOTE: was `#[repr(C, packed)]`; every field is a u64 / fn-pointer (all
// 8-byte aligned), so packed changes NOTHING about the layout but makes every
// field access an unaligned-reference hazard (E0793) — e.g. `state.x3` passed
// to `error!`, or `(*state).spsel` in `debug_assert_eq!`. `#[repr(C)]` gives
// the identical layout (verified by the offset_of! asserts below) without the
// hazard. The asm coupling relies on those offset asserts, not on packing.
#[repr(C)]
pub(crate) struct State {
	/// Stack selector
	pub spsel: u64,
	/// Exception Link Register
	pub elr_el1: extern "C" fn(extern "C" fn(usize), usize) -> !,
	/// Program Status Register
	pub spsr_el1: u64,
	/// User-level stack
	pub sp_el0: u64,
	/// Thread ID Register
	pub tpidr_el0: u64,
	/// X0 register
	pub x0: u64,
	/// X1 register
	pub x1: u64,
	/// X2 register
	pub x2: u64,
	/// X3 register
	pub x3: u64,
	/// X4 register
	pub x4: u64,
	/// X5 register
	pub x5: u64,
	/// X6 register
	pub x6: u64,
	/// X7 register
	pub x7: u64,
	/// X8 register
	pub x8: u64,
	/// X9 register
	pub x9: u64,
	/// X10 register
	pub x10: u64,
	/// X11 register
	pub x11: u64,
	/// X12 register
	pub x12: u64,
	/// X13 register
	pub x13: u64,
	/// X14 register
	pub x14: u64,
	/// X15 register
	pub x15: u64,
	/// X16 register
	pub x16: u64,
	/// X17 register
	pub x17: u64,
	/// X18 register
	pub x18: u64,
	/// X19 register
	pub x19: u64,
	/// X20 register
	pub x20: u64,
	/// X21 register
	pub x21: u64,
	/// X22 register
	pub x22: u64,
	/// X23 register
	pub x23: u64,
	/// X24 register
	pub x24: u64,
	/// X25 register
	pub x25: u64,
	/// X26 register
	pub x26: u64,
	/// X27 register
	pub x27: u64,
	/// X28 register
	pub x28: u64,
	/// X29 register
	pub x29: u64,
	/// X30 register
	pub x30: u64,
}

// === asm↔Rust offset coupling (start.s trap_exit / D4 tail) ===
// trap_exit restores registers via consecutive ldp pairs; reordering State
// fields silently shifts which value lands in which register.
const _: () = assert!(
	core::mem::offset_of!(State, sp_el0) == 24,
	"State.sp_el0 must be at offset 24 (trap_exit 2nd ldp pair, 2nd element)"
);
const _: () = assert!(
	core::mem::offset_of!(State, spsr_el1) == 16,
	"State.spsr_el1 must be at offset 16 (trap_exit 2nd ldp pair, 1st element)"
);
const _: () = assert!(
	core::mem::offset_of!(State, elr_el1) == 8,
	"State.elr_el1 must be at offset 8 (trap_exit 1st ldp pair, 2nd element)"
);
const _: () = assert!(
	core::mem::offset_of!(State, tpidr_el0) == 32,
	"State.tpidr_el0 must be at offset 32 (trap_exit 3rd ldp pair, 1st element)"
);
const _: () = assert!(
	core::mem::offset_of!(State, x21) == 0xd0,
	"State.x21 must be at offset 0xd0 (D4 tail hardcodes [x22, #0xd0])"
);
const _: () = assert!(
	core::mem::offset_of!(State, x22) == 0xd8,
	"State.x22 must be at offset 0xd8 (D4 tail hardcodes [x22, #0xd8])"
);
const _: () = assert!(
	core::mem::offset_of!(State, x0) == 40,
	"State.x0 must be at offset 40 (trap_exit 1st ldp pair, 2nd element reads x0 from [sp])"
);
const _: () = assert!(
	core::mem::offset_of!(State, x30) == 0x118,
	"State.x30 must be at offset 0x118 (x0@40 + 30*8 = 280; frame size 288)"
);
const _: () = assert!(
	core::mem::size_of::<State>() == 288,
	"State must be exactly 288 bytes (18 ldp pairs in trap_exit; asm hardcodes #288)"
);

pub struct BootStack {
	/// Stack for kernel tasks
	stack: VirtAddr,
}

pub struct CommonStack {
	/// Start address of allocated virtual memory region
	virt_addr: VirtAddr,
	/// Start address of allocated virtual memory region
	phys_addr: PhysAddr,
	/// Total size of all stacks
	total_size: usize,
}

pub enum TaskStacks {
	Boot(BootStack),
	Common(CommonStack),
}

impl TaskStacks {
	/// Size of the debug marker at the very top of each stack.
	///
	/// We have a marker at the very top of the stack for debugging (`0xdeadbeef`), which should not be overridden.
	pub const MARKER_SIZE: usize = 0x10;

	pub fn new(size: usize) -> Self {
		let user_stack_size = if size < KERNEL_STACK_SIZE {
			KERNEL_STACK_SIZE
		} else {
			size.align_up(BasePageSize::SIZE as usize)
		};
		let total_size = user_stack_size + DEFAULT_STACK_SIZE;
		let layout = PageLayout::from_size(total_size + 3 * BasePageSize::SIZE as usize).unwrap();
		let page_range = PageAlloc::allocate(layout).unwrap();
		let virt_addr = VirtAddr::from(page_range.start());
		let frame_layout = PageLayout::from_size(total_size).unwrap();
		let frame_range = FrameAlloc::allocate(frame_layout)
			.expect("Failed to allocate Physical Memory for TaskStacks");
		let phys_addr = PhysAddr::from(frame_range.start());

		let mut flags = PageTableEntryFlags::empty();
		flags.normal().writable().execute_disable();

		// map kernel stack into the address space
		crate::arch::mm::paging::map::<BasePageSize>(
			virt_addr + BasePageSize::SIZE,
			phys_addr,
			DEFAULT_STACK_SIZE / BasePageSize::SIZE as usize,
			flags,
		);

		// map user stack into the address space
		crate::arch::mm::paging::map::<BasePageSize>(
			virt_addr + DEFAULT_STACK_SIZE + 2 * BasePageSize::SIZE,
			phys_addr + DEFAULT_STACK_SIZE,
			user_stack_size / BasePageSize::SIZE as usize,
			flags,
		);

		// clear user stack — word-by-word to avoid compiler_builtins memset overflow
		let user_stack_va = virt_addr + DEFAULT_STACK_SIZE + 2 * BasePageSize::SIZE;
		warn!(
			"[TRACE-STACKS] clear user stack: va={:#x} size={:#x} ({} bytes)",
			user_stack_va.as_u64(),
			user_stack_size,
			user_stack_size
		);
		unsafe {
			let ptr = user_stack_va.as_mut_ptr::<u64>();
			let nwords = user_stack_size / size_of::<u64>();
			let words = core::slice::from_raw_parts_mut(ptr, nwords);
			for word in words.iter_mut() {
				*word = 0;
			}
		}
		warn!("[TRACE-STACKS] user stack clear done");

		TaskStacks::Common(CommonStack {
			virt_addr,
			phys_addr,
			total_size,
		})
	}

	pub fn from_boot_stacks() -> TaskStacks {
		let stack = VirtAddr::from_ptr(CURRENT_STACK_ADDRESS.load(Ordering::Relaxed));
		debug!("Using boot stack {stack:p}");

		TaskStacks::Boot(BootStack { stack })
	}

	pub fn get_user_stack_size(&self) -> usize {
		match self {
			TaskStacks::Boot(_) => 0,
			TaskStacks::Common(stacks) => stacks.total_size - DEFAULT_STACK_SIZE,
		}
	}

	pub fn get_user_stack(&self) -> VirtAddr {
		match self {
			TaskStacks::Boot(_) => VirtAddr::zero(),
			TaskStacks::Common(stacks) => {
				stacks.virt_addr + DEFAULT_STACK_SIZE + 2 * BasePageSize::SIZE
			}
		}
	}

	pub fn get_kernel_stack(&self) -> VirtAddr {
		match self {
			TaskStacks::Boot(stacks) => stacks.stack,
			TaskStacks::Common(stacks) => stacks.virt_addr + BasePageSize::SIZE,
		}
	}

	pub fn get_kernel_stack_size(&self) -> usize {
		match self {
			TaskStacks::Boot(_) => KERNEL_STACK_SIZE,
			TaskStacks::Common(_) => DEFAULT_STACK_SIZE,
		}
	}
}

impl Drop for TaskStacks {
	fn drop(&mut self) {
		// we should never deallocate a boot stack
		match self {
			TaskStacks::Boot(_) => {}
			TaskStacks::Common(stacks) => {
				debug!(
					"Deallocating stacks at {:p} with a size of {} KB",
					stacks.virt_addr,
					stacks.total_size >> 10,
				);

				crate::arch::mm::paging::unmap::<BasePageSize>(
					stacks.virt_addr,
					stacks.total_size / BasePageSize::SIZE as usize + 3,
				);
				let range = PageRange::from_start_len(
					stacks.virt_addr.as_usize(),
					stacks.total_size + 3 * BasePageSize::SIZE as usize,
				)
				.unwrap();
				unsafe {
					PageAlloc::deallocate(range);
				}

				let range =
					PageRange::from_start_len(stacks.phys_addr.as_usize(), stacks.total_size)
						.unwrap();
				unsafe {
					FrameAlloc::deallocate(range);
				}
			}
		}
	}
}

/*
 * https://fuchsia.dev/fuchsia-src/development/kernel/threads/tls and
 * and https://uclibc.org/docs/tls.pdf is used to understand variant 1
 * of the TLS implementations.
 */

extern "C" fn thread_exit(status: i32) -> ! {
	debug!("Exit thread with error code {status}!");
	core_scheduler().exit(status)
}

/// Static trace buffer for GDB inspection: task_start dumps registers here.
/// Layout: [0]=x0(func), [1]=x1(arg), [2]=x25, [3]=x30, [4]=SP after spsel,
///         [5]=SP after blr (i.e. SP at func entry), [8]=magic
#[unsafe(no_mangle)]
pub(crate) static mut TASK_START_TRACE: [u64; 16] = [0u64; 16];

#[unsafe(naked)]
extern "C" fn task_start(_f: extern "C" fn(usize), _arg: usize) -> ! {
	// `f` is in the `x0` register
	// `arg` is in the `x1` register

	naked_asm!(
		// === TRACE: dump register state to static buffer ===
		"adrp x8, {trace}",
		"add  x8, x8, #:lo12:{trace}",
		"str  x0, [x8, #0]",       // trace[0] = x0 (func)
		"str  x1, [x8, #8]",       // trace[1] = x1 (arg)
		"str  x25, [x8, #16]",     // trace[2] = x25
		"str  x30, [x8, #24]",     // trace[3] = x30 (LR from trap_exit)
		"mov  x9, #0x42",
		"str  x9, [x8, #64]",      // trace[8] = 0x42 (magic: task_start reached)
		// === end trace ===
		"msr spsel, {l0}",
		"mov x9, sp",
		"str  x9, [x8, #32]",      // trace[4] = SP after spsel (should be SP_EL0 = kernel_stack_top-16)
		"mov x25, x0",
		"mov x0, x1",
		"blr x25",
		// If func returns, we land here. Dump SP to verify stack integrity.
		"mov x9, sp",
		"str  x9, [x8, #40]",      // trace[5] = SP after func returns (should be back near kernel_stack_top-16)
		"mov x0, xzr",
		"adrp x4, {exit}",
		"add x4, x4, #:lo12:{exit}",
		"br x4",
		l0 = const 0,
		exit = sym thread_exit,
		trace = sym TASK_START_TRACE,
	)
}

impl TaskFrame for Task {
	fn create_stack_frame(&mut self, func: unsafe extern "C" fn(usize), arg: usize) {
		// Check if TLS is allocated already and if the task uses thread-local storage.
		#[cfg(not(feature = "common-os"))]
		if self.tls.is_none() {
			use crate::scheduler::task::tls::Tls;
			self.tls = Tls::from_env();
		}
		unsafe {
			// Set a marker for debugging at the very top.
			let mut stack = self.stacks.get_kernel_stack() + self.stacks.get_kernel_stack_size()
				- TaskStacks::MARKER_SIZE;
			*stack.as_mut_ptr::<u64>() = 0xdead_beefu64;

			// === SENTINEL FILL (Bug #3 diagnostic) ===
			// Bug #3 is a `ret`-to-0 (PC=0, x30=0, EC=0x21) after "Jumping into
			// application": a statically-linked libstdc++/LIEF `.init_array`
			// constructor performs an oversized write that lands INSIDE the
			// mapped kernel stack and zeroes a saved return-address slot. No
			// guard page catches it (that would be a Data Abort, EC=0x96...),
			// so we cannot see it as an overflow. To pin the clobbered slot we
			// pre-fill the entire kernel-stack BODY with an address-encoding
			// sentinel: word at VA `w` holds `0x5EED_0000_0000_0000 | (w & mask)`.
			// After a crash the fault dumper can classify each word:
			//   - value == expected_sentinel(addr)  -> untouched
			//   - value == 0                          -> zeroed by the bad write
			//   - value is a 0x407b.._0x4134.. text addr in a sentinel slot
			//                                         -> a real LR displaced here
			//   - anything else                       -> foreign write
			// The lowest address whose sentinel is broken marks the write's
			// start; the highest marks its end -> footprint + target slot.
			{
				let body_lo = self.stacks.get_kernel_stack().as_u64();
				// Fill everything below where the State will live; the State
				// region + marker are written explicitly afterwards.
				let body_hi = stack.as_u64() - size_of::<State>() as u64;
				let mut w = body_lo;
				while w < body_hi {
					// noalias &mut store (never merged into a memset).
					let slot = &mut *(w as *mut u64);
					*slot = 0x5eed_0000_0000_0000u64 | (w & 0x0000_ffff_ffff_fff8u64);
					w += 8;
				}
			}

			// Put the State structure expected by the ASM switch() function on the stack.
			stack -= size_of::<State>();

			let state = stack.as_mut_ptr::<State>();

			// write_bytes(state, 0, 288) causes LLVM to generate a memset that
			// overflows into the guard page at kernel_stack_top. The &mut reference
			// from slice::from_raw_parts_mut prevents the merge.
			let nwords = size_of::<State>() / size_of::<u64>();
			let state_words = core::slice::from_raw_parts_mut(state as *mut u64, nwords);
			for word in state_words.iter_mut() {
				*word = 0;
			}
			#[cfg(not(feature = "common-os"))]
			if let Some(tls) = &self.tls {
				(*state).tpidr_el0 = tls.thread_ptr().expose_provenance() as u64;
			}

			// The elr_el1 needs to hold the address of the
			// first function to be called when returning from exception handler.
			(*state).elr_el1 = task_start;
			(*state).x0 = func as usize as u64; // use second argument to transfer the entry point
			(*state).x1 = arg as u64;
			// EL1t (Option D, §1.2 / §2a.3): run tasks at EL1t so the CPU's
			// automatic SP->SP_EL1 switch lands exception frames on the per-core
			// exception stack E. SP_EL0 holds the task's KERNEL-stack top; SP_EL1
			// (E) is set at boot + restored by the trap_exit D4 tail.
			(*state).spsel = 0;
			/* Zero the condition flags; M[3:0]=0b0100 = EL1t, SPSEL=0. */
			(*state).spsr_el1 = 0x3e4;
			// §1C/§1D: verify the EL1t constants we just set. State is
			// #[repr(C, packed)], so read fields via addr_of!().read_unaligned()
			// (taking a field reference would be an unaligned-reference error).
			let got_spsel = {
				let p = core::ptr::addr_of!((*state).spsel);
				p.read_unaligned()
			};
			debug_assert_eq!(
				got_spsel, 0,
				"create_stack_frame: spsel must be 0 (EL1t) for hardware SP_EL1 auto-switch"
			);
			let got_spsr = {
				let p = core::ptr::addr_of!((*state).spsr_el1);
				p.read_unaligned()
			};
			debug_assert_eq!(
				got_spsr, 0x3e4,
				"create_stack_frame: spsr_el1 must be 0x3e4 (EL1t, SPSEL=0, FPU trap, DAIF masked)"
			);

			// Set the task's stack pointer entry to the stack we have just crafted.

			self.last_stack_pointer = stack;

			// §5A: verify frame lies within the mapped kernel stack region.
			{
				let kstack_lo = self.stacks.get_kernel_stack().as_u64();
				let kstack_hi = kstack_lo + self.stacks.get_kernel_stack_size() as u64;
				debug_assert!(
					stack.as_u64() >= kstack_lo
						&& stack.as_u64() + size_of::<State>() as u64 <= kstack_hi,
					"create_stack_frame: frame [{:#x}, {:#x}) outside kernel stack [{:#x}, {:#x})",
					stack.as_u64(),
					stack.as_u64() + size_of::<State>() as u64,
					kstack_lo,
					kstack_hi
				);
			}

			// EL1t: SP_EL0 holds the task body's stack pointer.
			//
			// REGRESSION FIX (2026-07-23, stopgap "a"): the previous commit
			// moved sp_el0 from the 1 MiB user stack to the DEFAULT_STACK_SIZE
			// (128 KiB) kernel region. The C++ static-init path
			// (_GLOBAL__sub_I__ZN9__gnu_cxx9__freeres_ev: getenv/_findenv_r/
			// strchr/strtoul + libstdc++ eh_alloc 0x12000 temp) overflows 128 KiB
			// under a debug build, clobbering a return slot -> EC=0x21. For a
			// `Common` stack (the init task) we therefore run the task body on
			// the full user-stack region (>= 1 MiB), restoring the pre-regression
			// headroom. The proper fix is Option D F-class C (dedicated static-init
			// stack, docs/option-d-stack-switch-design.md §7.4 / write-bytes doc §4.0.6).
			// `Boot` tasks keep the kernel-stack top (their only stack).
			//
			// kernel_stack_top = frame_base + STATE_SIZE + MARKER_SIZE.
			// Point SP_EL0 one push *below* the chosen top (top - 16) so the task
			// prologue's first `stp x29,x30,[sp,#-16]!` lands in the mapped
			// stack page, not the unmapped GUARD page at top+0 (Option D §11.8).
			let sp_top = match &self.stacks {
				TaskStacks::Boot(_) => {
					stack.as_u64()
						+ size_of::<State>() as u64
						+ TaskStacks::MARKER_SIZE as u64
				}
				TaskStacks::Common(_) => {
					// Use the top of the (much larger) user-stack region.
					let user_top = self.stacks.get_user_stack().as_u64()
						+ self.stacks.get_user_stack_size() as u64
						- TaskStacks::MARKER_SIZE as u64;
					user_top
				}
			};
			(*state).sp_el0 = sp_top - 16;

			// §5B: verify sp_el0 lies within the mapped user stack region (Common).
			if let TaskStacks::Common(_) = &self.stacks {
				let ustack_lo = self.stacks.get_user_stack().as_u64();
				let ustack_hi = ustack_lo + self.stacks.get_user_stack_size() as u64;
				let sp_el0_val = sp_top - 16;
				debug_assert!(
					sp_el0_val >= ustack_lo && sp_el0_val < ustack_hi,
					"create_stack_frame: sp_el0 {sp_el0_val:#x} outside user stack [{ustack_lo:#x}, {ustack_hi:#x})"
				);
			}
			// §5C: verify frame + marker don't overflow into the guard page.
			{
				let frame_end = stack.as_u64() + size_of::<State>() as u64;
				let guard_start = self.stacks.get_kernel_stack().as_u64()
					+ self.stacks.get_kernel_stack_size() as u64;
				debug_assert!(
					frame_end + TaskStacks::MARKER_SIZE as u64 <= guard_start,
					"create_stack_frame: frame+marker ({:#x}) overflows into guard ({:#x})",
					frame_end + TaskStacks::MARKER_SIZE as u64,
					guard_start
				);
			}

			// [V1] verification trace — idle-el1t-switch-redesign.md §6.
			// Placed AFTER the sp_el0 assignment (line above) so it captures the
			// TRUE pristine value, not the zeroed slot. Logs frame_base,
			// kernel_stack_top, size_of::<State>(), and the pristine sp_el0.
			// size_of_State != 288 => H-a3 (off-by size). sp_el0_slot == 0 =>
			// pristine frame wrong (create_stack_frame defect).
			// State is #[repr(C, packed)] — read the field via a raw pointer
			// with read_unaligned to avoid an unaligned reference (E0793).
			let v1_sp_el0 = core::ptr::addr_of!((*state).sp_el0).read_unaligned();
			warn!(
				"[V1] CREATE frame_base={:#x} ktop={:#x} size_of_State={} sp_el0_slot={:#x}",
				stack.as_u64(),
				self.stacks.get_kernel_stack().as_u64()
					+ self.stacks.get_kernel_stack_size() as u64,
				size_of::<State>(),
				v1_sp_el0,
			);

			// user_stack_pointer still populated for any legacy reader; the
			// EL1t model does not run the task body on the user stack.
			self.user_stack_pointer = self.stacks.get_user_stack()
				+ self.stacks.get_user_stack_size()
				- TaskStacks::MARKER_SIZE;

			*self.user_stack_pointer.as_mut_ptr::<u64>() = 0xdead_beefu64;
		}
	}
}

#[unsafe(no_mangle)]
pub(crate) extern "C" fn get_last_stack_pointer() -> u64 {
	// Trap next FPU instruction so we can lazily restore FPU state
	CPACR_EL1.modify(CPACR_EL1::FPEN::TrapEl0El1);
	isb(SY);
	let sp = core_scheduler().get_last_stack_pointer().as_u64();
	// [V3] verification trace — idle-el1t-switch-redesign.md §6. Fires on
	// EVERY idle switch-IN (this fn is called on every switch). Shows the
	// POST-D3 frame idle will resume from. Valid for several cycles then bad
	// => recent corruption window; bad from the FIRST switch-IN => pristine
	// frame wrong (H-a3 / create_stack_frame defect). Pins when idle.lsp was
	// first corrupted.
	{
		let cs = core_scheduler();
		if cs.get_current_task_id() == TaskId::from(0) {
			// current task being switched to IS idle: sp == idle frame base
			let slot = sp as *const u64;
			warn!(
				"[V3] idle selected: frame_base={:#x} sp_el0={:#x} elr={:#x}",
				sp,
				unsafe { *slot.add(3) },
				unsafe { *slot.add(1) },
			);
		}
	}
	// NOTE: do NOT fatal on 0 here. The early-boot/idle task (TaskStacks::Boot)
	// legitimately has last_stack_pointer == 0; callers that copy the frame
	// (D3) guard with cbz. A 0 here only matters on the real switch path
	// (mov sp, x0), which will fault visibly if truly bogus.
	if sp == 0 {
		debug!("get_last_stack_pointer() == 0 (early-boot/idle; expected)");
	}
	sp
}
