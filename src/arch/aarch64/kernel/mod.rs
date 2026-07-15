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

	// Allocate stack for the CPU and pass the addresses.
	let layout = Layout::from_size_align(KERNEL_STACK_SIZE, BasePageSize::SIZE as usize).unwrap();
	let stack = unsafe { alloc(layout) };
	assert!(!stack.is_null());
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
