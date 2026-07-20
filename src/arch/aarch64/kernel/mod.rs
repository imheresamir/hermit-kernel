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

use alloc::alloc::alloc;
use core::alloc::Layout;
use core::arch::global_asm;
use core::ptr;
use core::sync::atomic::{AtomicPtr, AtomicU32, Ordering};

pub(crate) use self::interrupts::wakeup_core;
pub(crate) use self::processor::set_oneshot_timer;
use crate::arch::aarch64::kernel::core_local::*;
use crate::arch::aarch64::mm::paging::{BasePageSize, PageSize};
use crate::config::*;

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
pub(crate) static CURRENT_STACK_ADDRESS: AtomicPtr<u8> = AtomicPtr::new(ptr::null_mut());

/// Early SMP release flag.
///
/// Secondary cores spin in `_start` until the boot core sets this to 1.
/// Placed in `.early_data` so it is mapped before paging is fully initialized.
#[unsafe(link_section = ".early_data")]
pub(crate) static EARLY_SMP_RELEASE: AlignedAtomicU32 = AlignedAtomicU32(AtomicU32::new(0));

	/// Emergency exception stack pool used very early on aarch64.
	///
	/// Design:
	/// - **Early boot** uses a small fixed pool of stacks in `.early_data` that is
	///   guaranteed mapped. This avoids recursive exception storms if `sp` is invalid.
	/// - **Later boot** should switch to real per-core exception stacks stored in
	///   core-local state once the allocator/paging are fully initialized.
	///
	/// For now we keep only the early pool (64 stacks). This keeps `.early_data`
	/// bounded and avoids scaling the image size linearly with core count.
	pub(crate) const HERMIT_EARLY_EXCEPTION_STACK_SIZE: usize = 64 * 1024;
	pub(crate) const HERMIT_EARLY_EXCEPTION_STACK_POOL_SIZE: usize = 64;

	#[unsafe(link_section = ".early_data")]
	#[unsafe(no_mangle)]
	pub(crate) static mut HERMIT_EARLY_EXCEPTION_STACK_POOL: [u8; HERMIT_EARLY_EXCEPTION_STACK_POOL_SIZE * HERMIT_EARLY_EXCEPTION_STACK_SIZE] =
		[0; HERMIT_EARLY_EXCEPTION_STACK_POOL_SIZE * HERMIT_EARLY_EXCEPTION_STACK_SIZE];

/// Unmap the per-core stack guard pages so a stack overflow faults (translation
/// fault → el1_sync prints ESR/FAR/ELR) instead of silently corrupting the
/// adjacent stack section.
///
/// Each `.X_stacks` section in `link.x` holds one slot per core, sized to the
/// matching `config.rs` constant, followed by a 4 KiB guard tail. The LIEF
/// patcher grows each section at deploy time to `N × (STACK_SIZE + GUARD_SIZE)`
/// for `N` cores detected at boot. `protect_stack_guards` walks `i in 0..N` for
/// each section and unmaps the guard page at `base + i*(STACK+GUARD) + STACK`.
///
/// `N` comes from `get_possible_cpus()` (the device tree), so this covers all
/// cores regardless of bring-up order. The kernel page tables must be live
/// (called from `boot_processor_init` after `mm::init()`) and it must run
/// before any task uses a stack.
///
/// The section base addresses come from the linker-provided `__start_X` symbols,
/// defined with `= .` inside each section. These carry a real (non-zero) link
/// address and are rebased by the loader's `R_AARCH64_RELATIVE` relocation — the
/// same path as `executable_start`. (An earlier `PROVIDE(__start_X = ADDR(.X))`
/// form evaluated `ADDR()` to 0 inside the `INSERT AFTER .tbss` script, so
/// `&__start_X` resolved to 0 at runtime and the unmap hit garbage addresses.)
pub(crate) fn protect_stack_guards() {
    use crate::arch::aarch64::mm::paging::unmap;
    use crate::config::{DEFAULT_STACK_SIZE, KERNEL_STACK_SIZE};
    use memory_addresses::VirtAddr;

    // Linker-provided section base symbols. These are defined with `= .` inside
    // each `.X_stacks` section in crates/rs6/link.x, so the linker gives them a
    // real (non-zero) link address and emits an `R_AARCH64_RELATIVE` relocation
    // that hermit-loader rebases by the load bias (just like `executable_start`).
    // The kernel is PIE, so we read the *runtime* address via the relocated
    // reference — no hardcoded offsets, no ELF walk.
    unsafe extern "C" {
        static __start_exception_stacks: u8;
        static __start_irq_stacks: u8;
        static __start_overflow_stacks: u8;
        static __start_task_stacks: u8;
        static __start_reactor_stacks: u8;
        static __start_idle_stacks: u8;
    }

    let section_bases: [usize; 6] = unsafe {
        [
            &__start_exception_stacks as *const u8 as usize,
            &__start_irq_stacks as *const u8 as usize,
            &__start_overflow_stacks as *const u8 as usize,
            &__start_task_stacks as *const u8 as usize,
            &__start_reactor_stacks as *const u8 as usize,
            &__start_idle_stacks as *const u8 as usize,
        ]
    };
    // Per-slot stack size from config.rs, parallel to `section_bases` above.
    let stacks: [usize; 6] = [
        DEFAULT_STACK_SIZE,
        KERNEL_STACK_SIZE,
        KERNEL_STACK_SIZE,
        DEFAULT_STACK_SIZE, // no config.rs const; placeholder
        DEFAULT_STACK_SIZE,
        KERNEL_STACK_SIZE,
    ];

    let n = detected_cores();
    let guard = BasePageSize::SIZE as u64;
    info!("protect_stack_guards: n={n} guard_page={guard:#x}");

    for i in 0..6 {
        let section_base = section_bases[i] as u64;
        let stack = stacks[i];
        info!("protect_stack_guards: section_base={section_base:#x} stack={stack:#x}");
        for j in 0..n {
            let guard_addr = section_base + j as u64 * (stack as u64 + guard) + stack as u64;
            let vaddr = VirtAddr::new(guard_addr);
            info!("protect_stack_guards: unmapping guard at {vaddr:p}");
            unmap::<BasePageSize>(vaddr, 1);
        }
    }
    info!("protect_stack_guards: done");
}

/// Number of cores whose per-core guards to unmap.
///
/// Reads the core count the kernel detects from the device tree (`get_possible_cpus`).
/// `protect_stack_guards` is called from `pre_init` only after `env::set_boot_info`
/// has made the FDT readable, so this is safe to call here. Never less than 1.
#[cfg(feature = "smp")]
fn detected_cores() -> usize {
	get_possible_cpus().max(1) as usize
}
#[cfg(not(feature = "smp"))]
fn detected_cores() -> usize {
	1
}

#[cfg(target_os = "none")]
global_asm!(include_str!("start.s"));

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

	// Unmap per-core stack guard pages so overflow faults instead of
	// corrupting adjacent sections. Must run AFTER `mm::init()` has made
	// Hermit's `L0TABLE_ADDRESS` page tables the active, populated root --
	// running it in `pre_init` (before paging init) would unmap from the
	// not-yet-active tables and the guard would be re-mapped when the real
	// tables are installed. The kernel page tables are shared, so unmapping
	// once on the boot core covers all cores.
	protect_stack_guards();

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

	// Use a static per-core slot in `.idle_stacks` instead of a heap allocation.
	// Slot N for core N: base + N * (KERNEL_STACK_SIZE + GUARD_SIZE). The guard
	// tail is unmapped by protect_stack_guards(), so an overflow faults instead of
	// corrupting the adjacent slot. CURRENT_STACK_ADDRESS points at the slot base;
	// start.s sets SP = base + KERNEL_STACK_SIZE (just below the guard).
	unsafe extern "C" {
		static __start_idle_stacks: u8;
	}
	let guard = BasePageSize::SIZE as usize;
	let slot = KERNEL_STACK_SIZE + guard;
	let core = core_id() as usize;
	let stack = unsafe { &__start_idle_stacks as *const u8 as usize + core * slot };
	CURRENT_STACK_ADDRESS.store(stack as *mut u8, Ordering::Relaxed);
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

		use memory_addresses::VirtAddr;

		use crate::arch::aarch64::kernel::start::{TTBR0, smp_start};
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
