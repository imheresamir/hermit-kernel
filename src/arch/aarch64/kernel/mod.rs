pub mod core_local;
pub mod interrupts;
#[cfg(feature = "kernel-stack")]
pub mod kernel_stack;
mod lscpu;
#[cfg(all(not(feature = "pci"), feature = "virtio"))]
pub mod mmio;
#[cfg(feature = "pci")]
pub mod pci;
pub mod processor;
pub mod scheduler;
pub mod serial;
#[cfg(target_os = "none")]
mod start;
pub mod systemtime;

use core::arch::global_asm;
use core::ptr;
use core::sync::atomic::{AtomicPtr, AtomicU32, Ordering};

use memory_addresses::VirtAddr;

pub(crate) use self::interrupts::wakeup_core;
pub(crate) use self::processor::set_oneshot_timer;
use crate::arch::aarch64::kernel::core_local::*;
use crate::arch::aarch64::mm::paging::{self, BasePageSize, PageSize};
use crate::config::*;

/// Unmap the 4 KB guard pages reserved (by `stacks_link.x`) between the stack
/// sections, so an overflowing stack faults in hardware instead of silently
/// corrupting the neighbour (the aarch64 CONFIG_VMAP_STACK model, "Option 2").
///
/// The guard regions are physically present in the loaded image (the loader
/// maps the whole RW segment), but we deliberately remove their page-table
/// entries here. `paging::unmap` (lower level) only writes an invalid PTE — it
/// does NOT free the backing frame, which is correct: these pages belong to the
/// kernel image, not the heap. (Do NOT use `crate::mm::unmap`, which panics on
/// an already-unmapped page and would deallocate image frames into the heap.)
///
/// Called once on the boot core after `mm::init()`, since all cores share the
/// TTBR0 root page table. Today each section holds a single (core 0) slot, so
/// there is exactly one guard page after `.exception_stacks` and one after
/// `.irq_stacks`. When the LIEF patcher grows a section to N cores it must
/// leave a guard page after every core's slot; this routine would then iterate
/// `core_count` slots per section (the per-slot stride is the section size).
unsafe fn protect_stack_guards() {
	// Unmap ONE `BasePageSize` guard page after each static stack section.
	//
	// NOTE: this deliberately unmaps a single 4 KB page, not the full 64 KB
	// `GUARD_SIZE` reserved by the linker. A prior attempt unmapped 16 pages
	// from each stack end and caused a silent boot hang at PCI scan: the 64 KB
	// regions overlapped early page-table pages in the kernel image, so the
	// ECAM MMIO read translated to the wrong PA and the device never answered.
	// Proper hardening is to unmap each `.stack_guard_*` section by its own
	// symbol (see STACK_GUARD_SIZE note), deferred until the linker gap layout
	// is reconciled. Single-page unmapping here is the proven-correct behavior.
	let guard_pages = 1usize;

	// Red-zone guard below the lowest stack (exception) — underflow trap.
	let lo_guard = (&raw const EXCEPTION_STACKS as *const u8 as usize) - STACK_GUARD_SIZE;
	paging::unmap::<BasePageSize>(VirtAddr::new(lo_guard as u64), guard_pages);
	debug!("Stack guard: unmapped red-zone below exception stack at {:#x} ({} pages)", lo_guard, guard_pages);

	// Guard page after .exception_stacks (single slot today).
	let exc_guard = (&raw const EXCEPTION_STACKS as *const u8 as usize) + EXCEPTION_STACK_SIZE;
	paging::unmap::<BasePageSize>(VirtAddr::new(exc_guard as u64), guard_pages);
	debug!("Stack guard: unmapped exception/irq gap at {:#x} ({} pages)", exc_guard, guard_pages);

	// Guard page after .irq_stacks (single slot today).
	let irq_guard = (&raw const IRQ_STACKS as *const u8 as usize) + IRQ_STACK_SIZE;
	paging::unmap::<BasePageSize>(VirtAddr::new(irq_guard as u64), guard_pages);
	debug!("Stack guard: unmapped irq/idle gap at {:#x} ({} pages)", irq_guard, guard_pages);

	// Tail guard after .idle_stacks.
	let idle_guard = (&raw const IDLE_STACKS as *const u8 as usize) + IDLE_STACK_SIZE;
	paging::unmap::<BasePageSize>(VirtAddr::new(idle_guard as u64), guard_pages);
	debug!("Stack guard: unmapped tail guard after idle stack at {:#x} ({} pages)", idle_guard, guard_pages);
}

#[repr(align(8))]
pub(crate) struct AlignedAtomicU32(AtomicU32);

/// `CPU_ONLINE` is the count of CPUs that finished initialization.
///
/// It also synchronizes initialization of CPU cores.
///
/// NOTE: This is read extremely early in the aarch64 entry stub (before paging
/// is fully initialized). Keep it in a dedicated section so it stays within the
/// early-mapped kernel image prefix even as the binary grows.
#[unsafe(link_section = ".early_data")]
pub(crate) static CPU_ONLINE: AlignedAtomicU32 = AlignedAtomicU32(AtomicU32::new(0));

/// Like `CPU_ONLINE`, this is accessed by the aarch64 entry stub before full
/// paging init. Keep it close to other early-boot state.
#[unsafe(link_section = ".early_data")]
#[unsafe(no_mangle)]
pub(crate) static CURRENT_STACK_ADDRESS: AtomicPtr<u8> = AtomicPtr::new(ptr::null_mut());

/// Early SMP release flag.
///
/// Secondary cores spin in `_start` until the boot core sets this to 1.
/// Placed in `.early_data` so it is mapped before paging is fully initialized.
#[unsafe(link_section = ".early_data")]
pub(crate) static EARLY_SMP_RELEASE: AlignedAtomicU32 = AlignedAtomicU32(AtomicU32::new(0));

/// Size of the unmapped guard page between adjacent static stack sections.
/// NOTE: keep this at ONE `BasePageSize` (4 KB). A previous attempt set it to
/// 0x10000 and made `protect_stack_guards()` unmap 16 pages per guard, which
/// unmapped 64 KB regions that overlapped early page-table pages allocated in
/// the kernel image and caused a silent boot hang at PCI scan (the ECAM MMIO
/// translated to the wrong PA). The correct hardening is to unmap each
/// `.stack_guard_*` section by its own symbol, not a fixed 64 KB from the
/// stack end — deferred until the linker gap layout is reconciled.
pub(crate) const STACK_GUARD_SIZE: usize = BasePageSize::SIZE as usize;

/// Per-core exception stack size. Expressed as a multiple of `BasePageSize`
/// so the per-core slot stride used in `start.s` (`core_id * size`) and the
/// guard unmapping are page-aligned for any configured base page size.
/// 16 pages (64 KB@4KB) — symmetric with `IRQ_STACK_SIZE`/`IDLE_STACK_SIZE`
/// and the aarch64 convention (Linux uses 64 KB for both IRQ and overflow
/// stacks). Sized UP from 4 KB -> 16 KB -> 32 KB; BOTH 16 KB and 32 KB
/// overflowed by ~80 B on a real client-connect fault (verified via ELF:
/// faulting FAR landed 0x50 past the section end with SP valid inside it,
/// at both sizes). The fault path is `do_sync` -> error log ->
/// `scheduler::abort()` which does a `kprintln!("Backtrace:")` frame walk up
/// `state.x29` — a deep path that needs > 32 KB. 64 KB gives a 32 KB margin.
/// This is a compile-time constant embedded in the assembly as an immediate.
pub(crate) const EXCEPTION_STACK_SIZE: usize = 16 * BasePageSize::SIZE as usize;

/// Per-core exception stacks in a dedicated ELF section.
///
/// Initially contains 1 stack (EXCEPTION_STACK_SIZE bytes). At deploy time,
/// LIEF grows this section to contain N stacks for an N-core deployment.
/// The assembly vector stubs index into this array by core ID.
///
/// Being the last section in the binary is critical: LIEF can grow it
/// by appending bytes without shifting any other sections.
#[repr(C, align(4096))]
pub(crate) struct ExceptionStacks([u8; EXCEPTION_STACK_SIZE]);

#[unsafe(link_section = ".exception_stacks")]
#[unsafe(no_mangle)]
pub(crate) static mut EXCEPTION_STACKS: ExceptionStacks =
	ExceptionStacks([0; EXCEPTION_STACK_SIZE]);

/// Per-core IRQ/FIQ stack size.
///
/// NOTE: this MUST be large enough for the deepest path that runs with
/// SP_EL1 == IRQ_STACKS[core_id]. The IRQ/FIQ handler body calls into Rust
/// (do_irq / do_fiq -> executor::run() -> handle_waiting_tasks() -> the
/// polled task future, e.g. the TCP listener setup), which can use far more
/// than a few KB of frames. The original design ran IRQs on the task's own
/// kernel stack (DEFAULT_STACK_SIZE = 64 KB); an undersized IRQ stack
/// overflows downward into .exception_stacks and silently corrupts the
/// task/kernel stack, causing a deterministic hang at the first heavy timer
/// IRQ. We therefore match the general kernel task stack size
/// (DEFAULT_STACK_SIZE = 64 KB) so the IRQ body has the same headroom the
/// original "IRQ runs on the task stack" design relied on.
pub(crate) const IRQ_STACK_SIZE: usize = DEFAULT_STACK_SIZE;

/// Per-core IRQ/FIQ stacks in a dedicated ELF section.
///
/// Each core's IRQ/FIQ handler runs on its own stack (Linux-aligned), so an
/// IRQ can never clobber the interrupted task's kernel stack, and a deep IRQ
/// cannot corrupt adjacent memory. The assembly switches SP_EL1 to
/// IRQ_STACKS[core_id] on every IRQ/FIQ entry and restores the interrupted
/// task's SP_EL1 on exit (via trap_exit).
///
/// Placed AFTER .exception_stacks (last in the RW segment) so LIEF can grow
/// all three stack sections independently without shifting earlier ones.
#[repr(C, align(4096))]
pub(crate) struct IrqStacks([u8; IRQ_STACK_SIZE]);

#[unsafe(link_section = ".irq_stacks")]
#[unsafe(no_mangle)]
pub(crate) static mut IRQ_STACKS: IrqStacks = IrqStacks([0; IRQ_STACK_SIZE]);

/// Per-core idle / first-task stack size.
///
/// This MUST match the general kernel task stack size (DEFAULT_STACK_SIZE =
/// 64 KB). The idle / first task is the *first* task
/// the kernel launches, and it runs before the general allocator is fully up,
/// so its stack cannot be allocated dynamically (you cannot dynamically
/// allocate the stack you boot the allocator on). It must therefore live in a
/// static section of the loaded image, mapped by the loader for its full size.
///
/// This is the third member of the "static per-core stack" family
/// (EXCEPTION_STACKS, IRQ_STACKS, IDLE_STACKS) — all three are in-image
/// sections grown by LIEF for an N-core deployment, eliminating dynamic stack
/// allocation entirely.
pub(crate) const IDLE_STACK_SIZE: usize = DEFAULT_STACK_SIZE;

/// Per-core idle / first-task stacks in a dedicated ELF section.
///
/// Each core's boot/idle task runs on its own stack (Linux-aligned), so the
/// first task never shares — and cannot corrupt — any other core's stack, and
/// the stack is mapped for its full `IDLE_STACK_SIZE` by the loader (no runtime
/// `alloc`/`map` needed). LIEF grows this section to N slots at deploy time.
///
/// Placed AFTER `.irq_stacks` (still last in the RW segment) so LIEF can grow
/// all three stack sections independently without shifting earlier ones.
#[repr(C, align(4096))]
pub(crate) struct IdleStacks([u8; IDLE_STACK_SIZE]);

#[unsafe(link_section = ".idle_stacks")]
#[unsafe(no_mangle)]
pub(crate) static mut IDLE_STACKS: IdleStacks = IdleStacks([0; IDLE_STACK_SIZE]);

/// Kernel configuration struct. Lives in `.early_data` so it's mapped early.
/// LIEF patches `core_count` at deploy time.
#[repr(C)]
pub(crate) struct KernelConfig {
	pub magic: u32,
	pub core_count: u32,
}

pub(crate) const KERNEL_CONFIG_MAGIC: u32 = 0x544c4b00;
pub(crate) const KERNEL_CONFIG_CORE_COUNT_OFFSET: u32 = 4;

/// Kernel configuration. Placed in `.early_data` for early mapping.
/// Assembly reads `core_count` to bounds-check the exception stack array.
#[unsafe(link_section = ".early_data")]
#[unsafe(no_mangle)]
pub(crate) static mut KERNEL_CONFIG: KernelConfig = KernelConfig {
	magic: KERNEL_CONFIG_MAGIC,
	core_count: 1,
};

#[cfg(target_os = "none")]
global_asm!(
	include_str!("start.s"),
	exception_stack_size = const EXCEPTION_STACK_SIZE,
	irq_stack_size = const IRQ_STACK_SIZE,
	kernel_config_offset = const KERNEL_CONFIG_CORE_COUNT_OFFSET,
);

#[cfg(feature = "smp")]
pub fn get_possible_cpus() -> u32 {
	let fdt = crate::env::fdt().unwrap();
	let cpu_count = fdt.cpus().count();
	u32::try_from(cpu_count).unwrap()
}

#[cfg(feature = "smp")]
pub fn get_processor_count() -> u32 {
	CPU_ONLINE.0.load(Ordering::Acquire)
}

#[cfg(not(feature = "smp"))]
pub fn get_processor_count() -> u32 {
	1
}

/// Real Boot Processor initialization as soon as we have put the first Welcome message on the screen.
#[cfg(target_os = "none")]
pub fn boot_processor_init() {
	processor::configure();

	crate::mm::init();
	crate::mm::print_information();
	// Unmap the 4 KB guard pages between the per-core stack sections so a
	// stack overflow faults in hardware (CONFIG_VMAP_STACK model) instead of
	// corrupting a neighbour. Must run after mm::init() installs the tables.
	unsafe {
		protect_stack_guards();
	}
	CoreLocal::get().add_irq_counter();
	interrupts::init();
	processor::detect_frequency();
	crate::logging::KERNEL_LOGGER.set_time(true);
	processor::print_information();
	systemtime::init();
	#[cfg(feature = "pci")]
	pci::init();

	finish_processor_init();

	// Safe-3 release point: core 0 has completed global init (paging, interrupts, etc)
	// and has installed its own per-core state. Now allow secondary cores to enter
	// the normal init path.
	EARLY_SMP_RELEASE.0.store(1, Ordering::Release);
	// Wake cores parked in `wfe`.
	unsafe { core::arch::asm!("sev", options(nostack, nomem)) };
}

/// Application Processor initialization
#[allow(dead_code)]
pub fn application_processor_init() {
	CoreLocal::install();
	interrupts::init_cpu();
	finish_processor_init();
}

fn finish_processor_init() {
	debug!("Initialized processor {}", core_id());

	// Point CURRENT_STACK_ADDRESS at this core's STATIC boot stack
	// (IDLE_STACKS[core_id]). No dynamic allocation: the idle / first task runs
	// before the general allocator is up, and its stack must already be mapped
	// as part of the loaded image (see IDLE_STACKS / Path C). The loader maps
	// the whole RW LOAD segment, so the per-core boot stack is valid from the
	// first instruction.
	//
	// This replaces the previous `alloc()`-based per-core stack, which was
	// never `map()`'d and faulted (level-1 translation fault) on the first
	// deep stack push of the launched task.
	let base = core_id() as usize * IDLE_STACK_SIZE;
	// SAFETY: IDLE_STACKS is a static [u8; IDLE_STACK_SIZE] in .idle_stacks;
	// take a raw pointer (no reference to the mutable static) and index it.
	let stack = unsafe { (&raw mut IDLE_STACKS.0).cast::<u8>().add(base) };
	CURRENT_STACK_ADDRESS.store(stack, Ordering::Relaxed);
}

pub fn boot_next_processor() {
	// This triggers to wake up the next processor (bare-metal/QEMU) or uhyve
	// to initialize the next processor.
	#[allow(unused_variables)]
	let cpu_online = CPU_ONLINE.0.fetch_add(1, Ordering::Release);

	#[allow(clippy::needless_return)]
	#[cfg(feature = "uhyve")]
	if crate::env::is_uhyve() {
		return;
	}

	#[cfg(all(target_os = "none", feature = "smp"))]
	if get_possible_cpus() > 1 {
		use core::arch::asm;
		use core::hint::spin_loop;

		use crate::arch::aarch64::kernel::start::{smp_start, TTBR0};
		use crate::mm::virtual_to_physical;

		if cpu_online == 0 {
			use aarch64_cpu::registers::{Readable, TTBR0_EL1};

			let virt_start = VirtAddr::from_ptr(smp_start as *const ());
			let phys_start = virtual_to_physical(virt_start).unwrap();
			assert!(virt_start.as_u64() == phys_start.as_u64());

			trace!("Virtual address of smp_start 0x{virt_start:x}");
			trace!("Physical address of smp_start 0x{phys_start:x}");

			let fdt = crate::env::fdt().unwrap();
			let psci_node = fdt.find_node("/psci").unwrap();

			let cpu_on = psci_node.property("cpu_on").unwrap().as_usize().unwrap();
			let cpu_on = u32::try_from(cpu_on).unwrap();
			trace!("CPU_ON: 0x{cpu_on:x}");

			let method = psci_node
				.property("method")
				.map(|node| node.as_str().unwrap())
				.unwrap_or("unknown");

			let ttbr0_addr = TTBR0_EL1.get();
			let ttbr0_ptr = ptr::with_exposed_provenance_mut(ttbr0_addr.try_into().unwrap());
			TTBR0.store(ttbr0_ptr, Ordering::Relaxed);

			for cpu_id in 1..get_possible_cpus() {
				debug!("Try to wake-up core {cpu_id}");

				if method == "hvc" {
					// call hypervisor to wakeup next core
					unsafe {
						asm!("hvc #0", in("x0") cpu_on, in("x1") cpu_id, in("x2") phys_start.as_u64(), in("x3") cpu_id, options(nomem, nostack));
					}
				} else if method == "smc" {
					// call secure monitor to wakeup next core
					unsafe {
						asm!("smc #0", in("x0") cpu_on, in("x1") cpu_id, in("x2") phys_start.as_u64(), in("x3") cpu_id, options(nomem, nostack));
					}
				} else {
					warn!("Method {method} isn't supported!");
					return;
				}

				// wait for next core
				while CPU_ONLINE.0.load(Ordering::Relaxed) < cpu_id + 1 {
					spin_loop();
				}
			}
		}
	}
}

pub fn print_statistics() {
	interrupts::print_statistics();
}
