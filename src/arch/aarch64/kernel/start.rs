#![allow(dead_code)]

use core::arch::{asm, naked_asm};
#[cfg(feature = "smp")]
use core::ptr;
#[cfg(feature = "smp")]
use core::sync::atomic::AtomicPtr;

use aarch64_cpu::asm::barrier::{SY, dsb};
use hermit_entry::Entry;
use hermit_entry::boot_info::RawBootInfo;

use crate::arch::aarch64::kernel::scheduler::TaskStacks;
use crate::config::{DEFAULT_STACK_SIZE, KERNEL_STACK_SIZE};
use crate::env;

// Per-core exception-stack base symbol (defined in link.x as `.exception_stacks`)
// — PIE-rebased by the loader, so `sym` yields the correct runtime address.
// Used by Phase 2a.1 to set SP_EL1 to this core's exception stack top (Option D).
unsafe extern "C" {
	static __start_exception_stacks: u8;
}

/*
 * Memory types available.
 */
#[allow(non_upper_case_globals)]
const MT_DEVICE_nGnRnE: u64 = 0;
#[allow(non_upper_case_globals)]
const MT_DEVICE_nGnRE: u64 = 1;
const MT_DEVICE_GRE: u64 = 2;
const MT_NORMAL_NC: u64 = 3;
const MT_NORMAL: u64 = 4;

/*
 * TCR flags
 */
const TCR_IRGN_WBWA: u64 = ((1) << 8) | ((1) << 24);
const TCR_ORGN_WBWA: u64 = ((1) << 10) | ((1) << 26);
const TCR_SHARED: u64 = ((3) << 12) | ((3) << 28);
const TCR_TBI0: u64 = 1 << 37;
const TCR_TBI1: u64 = 1 << 38;
const TCR_ASID16: u64 = 1 << 36;
const TCR_TG1_16K: u64 = 1 << 30;
const TCR_TG1_4K: u64 = 0 << 30;
const TCR_FLAGS: u64 = TCR_IRGN_WBWA | TCR_ORGN_WBWA | TCR_SHARED;

/// Number of virtual address bits for 4KB page
const VA_BITS: u64 = 48;

unsafe extern "C" {
	// NOTE: The build prefixes exported kernel symbols with `hermit_`.
	// We reference the final symbol name here so debuggers can set breakpoints by name.
	static hermit_vector_table: u8;
}

/// Entrypoint - Initialize Stack pointer and Exception Table
#[unsafe(no_mangle)]
#[unsafe(naked)]
pub unsafe extern "C" fn _start(boot_info: Option<&'static RawBootInfo>, cpu_id: u32) -> ! {
	// validate signatures
	// `_Start` is compatible to `Entry`
	{
		unsafe extern "C" fn _entry(_boot_info: &'static RawBootInfo, _cpu_id: u32) -> ! {
			unreachable!()
		}
		pub type _Start =
			unsafe extern "C" fn(boot_info: Option<&'static RawBootInfo>, cpu_id: u32) -> !;
		const _ENTRY: Entry = _entry;
		const _START: _Start = _start;
		const _PRE_INIT: _Start = pre_init;
	}

	naked_asm!(
		// Determine core id from MPIDR.
		"mrs x4, mpidr_el1",
		"and x4, x4, #0xff",

		// Only core 0 is allowed to run early boot. Park all other cores in a
		// low-power loop with interrupts masked until the boot core is ready to
		// bring them up.
		"cbz x4, 1f",
		"msr spsel, #1",
		"msr daifset, #0xf",
		"2:",
		"adrp x8, {early_smp_release}",
		"ldr w5, [x8, #:lo12:{early_smp_release}]",
		"cbnz w5, 3f",
		"wfe",
		"b 2b",
		"3:",
		// released: jump into the regular init path
		"b {pre_init}",
		"1:",

		// We want to use sp_el1 as early as possible.
		"msr spsel, #1",

		// Set exception vector base early so any fault before full init goes to
		// the kernel vector table (start.s) instead of the low 0x000.. vectors.
		// Requires a valid stack (we're on sp_el1 now).
		"adrp x9, {vt}",
		"add  x9, x9, #:lo12:{vt}",
		"msr  vbar_el1, x9",
		"isb",

		// Overwrite RSP if `CURRENT_STACK_ADDRESS != 0`
		"adrp x8, {current_stack_address}",
		"ldr x4, [x8, #:lo12:{current_stack_address}]",
		"cmp x4, 0",
		"b.eq 3f",
		"mov sp, x4",
		"b 4f",
		"3:",
		"mov x4, sp",
		"4:",
		"str x4, [x8, #:lo12:{current_stack_address}]",

		// Add stack top offset
		"mov x8, {stack_top_offset}",
		"add sp, sp, x8",

		// === Phase 2a.1: per-core exception-stack SP_EL1 (Option D) ===
		// spsel=1 (line above), so `sp` aliases SP_EL1; setting SP_EL1 == `mov sp`.
		// Tasks still run EL1h for now, so this only conditions the early/boot
		// exception path; the vector rewrite (2a.2) later relies on SP_EL1=E.
		"mrs x24, mpidr_el1",
		"and x24, x24, #0xff",       // core index (aff0; >64 cores via §7.2 port)
		"and x24, x24, #0x3f",
		"adrp x25, {exc_start}",
		"add  x25, x25, #:lo12:{exc_start}",
		"mov  x26, {exception_stack_size}", // DEFAULT_STACK_SIZE (0x20000) = MOVZ-able
		"add  x26, x26, #0x1000",         // + GUARD => slot_stride (STACK+GUARD)
		"mul  x24, x24, x26",
		"add  x25, x25, x24",        // base of this core's exception slot
		"mov  x26, {exception_stack_size}", // DEFAULT_STACK_SIZE (128KiB exception scratch stack)
		"add  x25, x25, x26",        // SP_EL1 = slot base + STACK = top of usable exception stack
		// NOTE: `msr sp_el1, x25` is UNDEFINED at EL1 (SP_EL1 is EL2/EL3-writable
		// only). At EL1h (spsel=1) `sp` aliases SP_EL1, so `mov sp, x25` is the
		// legal way to set it. Do NOT use `msr sp_el1`.
		"mov  sp, x25",

		// Jump to Rust code
		"b {pre_init}",

		early_smp_release = sym super::EARLY_SMP_RELEASE,
		stack_top_offset = const KERNEL_STACK_SIZE - TaskStacks::MARKER_SIZE,
		current_stack_address = sym super::CURRENT_STACK_ADDRESS,
		vt = sym hermit_vector_table,
		pre_init = sym pre_init,
		exc_start = sym __start_exception_stacks,
		exception_stack_size = const DEFAULT_STACK_SIZE,  // SP_EL1=E scratch stack (128KiB + GUARD per design §1.1)
	)
}

#[cfg(feature = "smp")]
const fn tcr_size(x: u64) -> u64 {
	((64 - x) << 16) | (64 - x)
}

#[cfg(feature = "smp")]
const fn mair(attr: u64, mt: u64) -> u64 {
	attr << (mt * 8)
}

#[cfg(feature = "smp")]
pub(crate) static TTBR0: AtomicPtr<u8> = AtomicPtr::new(ptr::null_mut());

#[cfg(all(target_os = "none", feature = "smp"))]
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn ap_trace_entry() {
	// Called from smp_start (assembly) right before jumping to pre_init,
	// to confirm the AP survived MMU/paging setup. cpu_id from MPIDR.
	use aarch64_cpu::registers::{Readable, MPIDR_EL1};
	let cpu_id = MPIDR_EL1.get() & 0xff;
	warn!("[TRACE-SMP] AP {cpu_id} smp_start setup done, entering pre_init");
}

#[cfg(feature = "smp")]
#[unsafe(naked)]
pub(crate) unsafe extern "C" fn smp_start() -> ! {
	// Prepare system control register (SCTRL)
	//
	// UCI     [26] Enables EL0 access in AArch64 for DC CVAU, DC CIVAC,
	//              DC CVAC and IC IVAU instructions
	// EE      [25] Explicit data accesses at EL1 and Stage 1 translation
	//              table walks at EL1 & EL0 are little-endian
	// EOE     [24] Explicit data accesses at EL0 are little-endian
	// WXN     [19] Regions with write permission are not forced to XN
	// nTWE    [18] WFE instructions are executed as normal
	// nTWI    [16] WFI instructions are executed as normal
	// UCT     [15] Enables EL0 access in AArch64 to the CTR_EL0 register
	// DZE     [14] Execution of the DC ZVA instruction is allowed at EL0
	// I       [12] Instruction caches enabled at EL0 and EL1
	// UMA     [9]  Disable access to the interrupt masks from EL0
	// SED     [8]  The SETEND instruction is available
	// ITD     [7]  The IT instruction functionality is available
	// THEE    [6]  ThumbEE is disabled
	// CP15BEN [5]  CP15 barrier operations disabled
	// SA0     [4]  Stack Alignment check for EL0 enabled
	// SA      [3]  Stack Alignment check enabled
	// C       [2]  Data and unified enabled
	// A       [1]  Alignment fault checking disabled
	// M       [0]  MMU enable
	#[cfg(target_endian = "little")]
	const SCTLR_EL1: u64 = 0b100_0000_0101_1101_0000_0001_1101;
	// The same, but EE and EOE are set to 1 for big endian.
	#[cfg(target_endian = "big")]
	const SCTLR_EL1: u64 = 0b111_0000_0101_1101_0000_0001_1101;

	naked_asm!(
		// disable interrupts
		"msr daifset, #0b111",

		// we want to use sp_el1!
		"msr spsel, #1",

		// Set exception vector base early (mirrors _start) so any fault in the
		// AP bring-up below goes to the kernel vector table (start.s) and prints
		// ESR/FAR/ELR instead of taking the low 0x000 vectors (unmapped) and
		// dying silently. This is the only way to see WHY a secondary core dies.
		"adrp x9, {vt}",
		"add  x9, x9, #:lo12:{vt}",
		"msr  vbar_el1, x9",
		"isb",

		// reset thread id registers
		"msr tpidr_el0, xzr",
		"msr tpidr_el1, xzr",

		// Disable the MMU and set the correct endianness
		// by either clearing or setting bits 25 and 24 (EE and EOE)
		"dsb sy",
		"mrs x2, sctlr_el1",
		"bic x2, x2, #0x1",
		#[cfg(target_endian = "little")]
		"bic x2, x2, #(1 << 24 | 1 << 25)",
		#[cfg(target_endian = "big")]
		"orr x2, x2, #(1 << 24 | 1 << 25)",
		"msr sctlr_el1, x2",
		"isb",

		"ic iallu",
		"tlbi vmalle1is",
		"dsb ish",

		// Setup memory attribute type tables
		"ldr x1, ={mair_el1}",
		"msr mair_el1, x1",

		// Setup translation control register (TCR)
		"mrs x0, id_aa64mmfr0_el1",
		"and x0, x0, 0xF",
		"lsl x0, x0, 32",
		"ldr x1, ={tcr_bits}",
		"orr x0, x0, x1",
		"mrs x1, id_aa64mmfr0_el1",
		"bfi x0, x1, #32, #3",
		"msr tcr_el1, x0",

		// Enable FP/ASIMD in Architectural Feature Access Control Register,
		"mov x0, 3",
		"lsl x0, x0, 20",
		"msr cpacr_el1, x0",

		// Reset debug control register
		"msr mdscr_el1, xzr",

		// Memory barrier
		"dsb sy",

		// Overwrite RSP if `CURRENT_STACK_ADDRESS != 0`
		"adrp x8, {current_stack_address}",
		"ldr x4, [x8, #:lo12:{current_stack_address}]",
		"cmp x4, 0",
		"b.eq 3f",
		"mov sp, x4",
		"b 4f",
		"3:",
		"mov x4, sp",
		"4:",
		"str x4, [x8, #:lo12:{current_stack_address}]",

		// Add stack top offset
		"mov x8, {stack_top_offset}",
		"add sp, sp, x8",

		// === Phase 2a.1: per-core exception-stack SP_EL1 (Option D) ===
		// spsel=1, so `sp` aliases SP_EL1; setting SP_EL1 == `mov sp`. Mirrors
		// the BSP _start block above (inserted after `add sp,sp,x8`).
		"mrs x24, mpidr_el1",
		"and x24, x24, #0xff",       // core index (aff0; >64 cores via §7.2 port)
		"and x24, x24, #0x3f",
		"adrp x25, {exc_start}",
		"add  x25, x25, #:lo12:{exc_start}",
		"mov  x26, {exception_stack_size}", // DEFAULT_STACK_SIZE (0x20000) = MOVZ-able
		"add  x26, x26, #0x1000",         // + GUARD => slot_stride (STACK+GUARD)
		"mul  x24, x24, x26",
		"add  x25, x25, x24",        // base of this core's exception slot
		"mov  x26, {exception_stack_size}", // DEFAULT_STACK_SIZE (128KiB exception scratch stack)
		"add  x25, x25, x26",        // SP_EL1 = slot base + STACK = top of usable exception stack
		// NOTE: `msr sp_el1, x25` is UNDEFINED at EL1 (SP_EL1 is EL2/EL3-writable
		// only). At EL1h (spsel=1) `sp` aliases SP_EL1, so `mov sp, x25` is the
		// legal way to set it. Do NOT use `msr sp_el1`.
		"mov  sp, x25",

		"msr ttbr1_el1, xzr",
		"adrp x8, {ttbr0}",
		"ldr x5, [x8, #:lo12:{ttbr0}]",
		"msr ttbr0_el1, x5",

		"ldr x0, ={sctlr_el1}",
		"msr sctlr_el1, x0",

		// initialize argument for pre_init
		"mov x0, xzr",
		"mrs x1, mpidr_el1",
		"and x1, x1, #0xff",

		// Jump to Rust code
		"bl {ap_trace_entry}",
		"b {pre_init}",

		mair_el1 = const mair(0x00, MT_DEVICE_nGnRnE) | mair(0x04, MT_DEVICE_nGnRE) | mair(0x0c, MT_DEVICE_GRE) | mair(0x44, MT_NORMAL_NC) | mair(0xff, MT_NORMAL),
		tcr_bits = const tcr_size(VA_BITS) | TCR_TG1_4K | TCR_FLAGS,
		stack_top_offset = const KERNEL_STACK_SIZE - TaskStacks::MARKER_SIZE,
		current_stack_address = sym super::CURRENT_STACK_ADDRESS,
		sctlr_el1 = const SCTLR_EL1,
		ttbr0 = sym TTBR0,
		pre_init = sym pre_init,
		exc_start = sym __start_exception_stacks,
		exception_stack_size = const DEFAULT_STACK_SIZE,  // SP_EL1=E scratch stack (128KiB + GUARD per design §1.1)
		ap_trace_entry = sym ap_trace_entry,
		vt = sym hermit_vector_table,
	)
}

#[inline(never)]
#[unsafe(no_mangle)]
unsafe extern "C" fn pre_init(boot_info: Option<&'static RawBootInfo>, cpu_id: u32) -> ! {
	let sp: u64;
	unsafe {
		core::arch::asm!("mov {0}, sp", out(reg) sp, options(nostack));
	}
	warn!("[TRACE-PRE-INIT] cpu_id={} sp={:#x}", cpu_id, sp);

	// set exception table
	unsafe {
		asm!(
			"adrp x4, {vector_table}",
			"add x4, x4, #:lo12:{vector_table}",
			"msr vbar_el1, x4",
			vector_table = sym hermit_vector_table,
			out("x4") _,
			options(nostack),
		);
	}

	// Memory barrier
	dsb(SY);

	if cpu_id == 0 {
		env::set_boot_info(*boot_info.unwrap());

		crate::boot_processor_main()
	} else {
		#[cfg(not(feature = "smp"))]
		{
			let style = anstyle::Style::new().fg_color(Some(anstyle::AnsiColor::Red.into()));
			let preamble = format_args!("[            ][{cpu_id}][{style}ERROR{style:#}]");
			println!(
				"{preamble} Secondary core booted, but Hermit was not built with SMP support!"
			);
			loop {
				crate::arch::kernel::processor::halt();
			}
		}
		#[cfg(feature = "smp")]
		crate::application_processor_main()
	}
}
