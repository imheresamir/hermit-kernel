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
pub mod slot_pool;
#[cfg(target_os = "none")]
mod start;
pub mod systemtime;

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
	use memory_addresses::VirtAddr;

	use crate::arch::aarch64::mm::paging::{get_page_table_entry, unmap};

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
		static __start_exception_slots: u8;
		static __end_exception_slots: u8;
	}

	// The per-task exception-slot pool (`.exception_slots`) is a NEW section the
	// hermit-loader's grow/rebase pass does not yet know about, so it maps
	// `.exception_stacks` (and the other stack sections) but leaves the slot
	// region with NO page-table entries. `&__start_exception_slots` is already
	// the correct *runtime* VA (the loader rebases it via R_AARCH64_RELATIVE,
	// like every other kernel symbol), but without PTEs trap_entry faults with
	// a translation abort the first time it writes a frame there. Map the whole
	// region RW (identity phys==virt, matching how the loader maps the rest of
	// the kernel image) before protect_stack_guards() unmaps the per-slot guard
	// tails. Slot contents are scratch (overwritten on first dispatch), so the
	// exact prior contents don't matter.
	{
		use memory_addresses::PhysAddr;

		use crate::arch::aarch64::mm::paging::{PageTableEntryFlags, map};

		let start = &raw const __start_exception_slots as usize;
		let end = &raw const __end_exception_slots as usize;
		let size = end.saturating_sub(start);
		if size > 0 {
			let pages = size.div_ceil(BasePageSize::SIZE as usize);
			let mut flags = PageTableEntryFlags::empty();
			flags.normal().writable().execute_disable();
			// SAFETY: `start`/`end` are linker-provided section bounds; mapping
			// this VA range identity is what the loader should have done.
			map::<BasePageSize>(
				VirtAddr::new(start as u64),
				PhysAddr::new(start as u64),
				pages,
				flags,
			);
			info!(
				"protect_stack_guards: mapped .exception_slots [{start:#x}, {end:#x}) = {pages} pages"
			);
		}
	}

	let section_bases: [usize; 7] = unsafe {
		[
			&__start_exception_stacks as *const u8 as usize,
			&__start_irq_stacks as *const u8 as usize,
			&__start_overflow_stacks as *const u8 as usize,
			&__start_task_stacks as *const u8 as usize,
			&__start_reactor_stacks as *const u8 as usize,
			&__start_idle_stacks as *const u8 as usize,
			&__start_exception_slots as *const u8 as usize,
		]
	};
	// Per-slot stack size from config.rs, parallel to `section_bases` above.
	// The exception stack is DEFAULT_STACK_SIZE (128KiB) -- a scratch stack for
	// trap_entry + dispatch (§1.1); deep handler work runs on the task's kernel
	// stack. All four sources (link.x, start.rs, core_local.rs, here) are 128KiB.
	let stacks: [usize; 7] = [
		DEFAULT_STACK_SIZE,
		KERNEL_STACK_SIZE,
		KERNEL_STACK_SIZE,
		DEFAULT_STACK_SIZE, // no config.rs const; placeholder
		DEFAULT_STACK_SIZE,
		KERNEL_STACK_SIZE,
		EXCEPTION_SLOT_SIZE, // per-task scratch slot
	];

	let n = max_bootable_cores();
	let guard = BasePageSize::SIZE as u64;
	info!("protect_stack_guards: n={n} guard_page={guard:#x}");

	for i in 0..7 {
		let section_base = section_bases[i] as u64;
		let stack = stacks[i];
		// The exception-slot section holds SLOTS_PER_CORE slots PER core
		// (stride = slot + guard), so its guard count is cores × SLOTS_PER_CORE.
		// Every other section has exactly one element per core.
		let elements = if i == 6 {
			n * SLOTS_PER_CORE
		} else {
			n
		};
		info!(
			"protect_stack_guards: section_base={section_base:#x} stack={stack:#x} elements={elements}"
		);
		for j in 0..elements {
			let guard_addr = section_base + j as u64 * (stack as u64 + guard) + stack as u64;
			let vaddr = VirtAddr::new(guard_addr);
			info!("protect_stack_guards: unmapping guard at {vaddr:p}");
			unmap::<BasePageSize>(vaddr, 1);
			// Runtime invariant net (R3): for the per-task slot section, the
			// State frame (slot_top - 288) must NOT share a page with the guard
			// we just unmapped. If the frame page is now unmapped, the guard
			// unmap zapped the frame -> the next exception faults. This would
			// have caught the 0x1200 (non-page-aligned) body bug at boot.
			if i == 6 {
				let frame_base = guard_addr - 288; // State frame size (288 B), pinned by config/slot_pool const-asserts
				let frame_page = frame_base & !0xfff;
				let guard_page = guard_addr & !0xfff;
				debug_assert_ne!(
					frame_page, guard_page,
					"slot frame page aliases guard page -- EXCEPTION_SLOT_SIZE not page-aligned"
				);
				debug_assert!(
					get_page_table_entry::<BasePageSize>(VirtAddr::new(frame_page)).is_some(),
					"slot frame page must remain MAPPED after guard unmap"
				);
				debug_assert!(
					get_page_table_entry::<BasePageSize>(VirtAddr::new(guard_page)).is_none(),
					"slot guard page must be UNMAPPED"
				);
			}
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

/// Maximum number of cores this kernel will actually bring up.
///
/// Number of cores the LINKER LAYOUT actually supports for the per-core stack
/// regions. There is deliberately NO `MAX_CORES` constant anywhere: the linker
/// lays out exactly one core's worth of stacks by default, and if a build
/// configures the layout for more cores (growing the `.exception_stacks` /
/// `.X_stacks` sections), the kernel OBSERVES that larger size here and
/// automatically supports more cores. We derive the supported count from the
/// *actual size* of the `.exception_stacks` section divided by the per-core
/// stride the boot path uses (start.rs `smp_start` SP_EL1 setup, core_local.rs
/// `e_top`): `DEFAULT_STACK_SIZE + GUARD`. This is the
/// "num_cpu_cores_supported_by_stacks_configured_by_linker" half of the rule.
///
/// SAFETY: `__start_exception_stacks` / `__end_exception_stacks` are absolute
/// symbols defined by link.x (INSERT AFTER .tbss). They bracket one contiguous
/// region; reading them as addresses is sound.
fn linker_supported_cores() -> usize {
	unsafe extern "C" {
		static __start_exception_stacks: u8;
		static __end_exception_stacks: u8;
	}
	let size = unsafe {
		(&__end_exception_stacks as *const u8 as usize)
			- (&__start_exception_stacks as *const u8 as usize)
	};
	// Per-core stride in start.rs `smp_start`: DEFAULT_STACK_SIZE (0x20000) + GUARD (0x1000).
	const PER_CORE_STRIDE: usize = 0x20000 + 0x1000;
	// Guard against a zero/garbage layout: always at least 1 core.
	size / PER_CORE_STRIDE.max(1)
}

/// Boot core count = `min(num_cpu_cores_available_from_fdt,
/// num_cpu_cores_supported_by_stacks_configured_by_linker)` — i.e. the number
/// of physical cores the device tree reports, clamped to the number of cores
/// the stack layout (linker) was configured to support. This is the
/// `min(available, supported_by_stacks_configured_by_linker)` rule: an
/// over-provisioned box (more physical cores than the link config sized for)
/// boots the supported subset; under-provisioned hardware boots every core it
/// has. Cores beyond the supported count are never PSCI-woken, so their
/// per-core SP_EL1 (which would index past the stack sections) is never
/// dereferenced — no crash.
#[cfg(feature = "smp")]
fn max_bootable_cores() -> usize {
	core::cmp::min(get_possible_cpus() as usize, linker_supported_cores()).max(1)
}
#[cfg(not(feature = "smp"))]
fn max_bootable_cores() -> usize {
	1
}

/// Number of cores this kernel will actually bring up this boot — i.e. the
/// count `boot_next_processor` PSCI-wakes (and the count `synch_all_cores`
/// must wait for). Equals `max_bootable_cores()`: the FDT core count clamped
/// to the linker-stack-supported count. This is the SINGLE source of truth
/// for "how many cores boot", shared by both the wake path and the barrier so
/// they can never disagree (a disagreement is exactly the `-smp N` hang:
/// the barrier waits for N physical cores while only the clamped subset booted).
#[cfg(feature = "smp")]
pub fn boot_core_count() -> u32 {
	max_bootable_cores() as u32
}
#[cfg(not(feature = "smp"))]
pub fn boot_core_count() -> u32 {
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
	warn!("[TRACE-SMP] AP application_processor_init: CoreLocal::install");
	CoreLocal::install();
	warn!("[TRACE-SMP] AP application_processor_init: interrupts::init_cpu");
	interrupts::init_cpu();
	warn!("[TRACE-SMP] AP application_processor_init: finish_processor_init");
	finish_processor_init();
}

fn finish_processor_init() {
	debug!("Initialized processor {}", core_id());

	// CURRENT_STACK_ADDRESS holds the *runtime* (rebased) base of the boot core's
	// idle stack, captured by _start from the loader's SP. Extend it per core so
	// each AP gets its own slot; for core 0 this is a no-op.
	//
	// Do NOT recompute from `&__start_idle_stacks`: that symbol resolves to the
	// image's `.idle_stacks` *section* (used only by protect_stack_guards() to
	// unmap per-core guard tails in the loaded image), which is a DIFFERENT VAS
	// from the loader's live boot stack where the idle task actually runs. The
	// DIAG in interrupts.rs confirms this: kernel_stack_top from the link-section
	// symbol is 0x4167d000, while the live fault sits at 0x800015d82000 — two
	// distinct addresses for "the idle stack".
	let base = CURRENT_STACK_ADDRESS.load(Ordering::Relaxed) as usize;
	let guard = BasePageSize::SIZE as usize;
	let slot = KERNEL_STACK_SIZE + guard;
	let core = core_id() as usize;
	let stack = base + core * slot;
	CURRENT_STACK_ADDRESS.store(stack as *mut u8, Ordering::Relaxed);
}

pub fn boot_next_processor() {
	warn!(
		"[TRACE-SMP] boot_next_processor entered, get_possible_cpus={}",
		get_possible_cpus()
	);
	// This triggers to wake up the next processor (bare-metal/QEMU) or uhyve
	// to initialize the next processor.
	#[allow(unused_variables)]
	let cpu_online = CPU_ONLINE.0.fetch_add(1, Ordering::Release);
	warn!("[TRACE-SMP] CPU_ONLINE now={}", CPU_ONLINE.0.load(Ordering::Relaxed));

	#[allow(clippy::needless_return)]
	#[cfg(feature = "uhyve")]
	if crate::env::is_uhyve() {
		return;
	}

	#[cfg(all(target_os = "none", feature = "smp"))]
	if max_bootable_cores() > 1 {
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

			let max_cores = max_bootable_cores();
			warn!(
				"[TRACE-SMP] max_bootable_cores={} (dt={}, linker_supported={})",
				max_cores,
				get_possible_cpus(),
				linker_supported_cores()
			);
			for cpu_id in 1..max_cores {
				warn!("[TRACE-SMP] Try to wake-up core {cpu_id} via method={method}");

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

				warn!(
					"[TRACE-SMP] core {cpu_id} wakeup sent, spinning for CPU_ONLINE>={}",
					cpu_id + 1
				);
				// wait for next core
				while CPU_ONLINE.0.load(Ordering::Relaxed) < (cpu_id as u32) + 1 {
					spin_loop();
				}
				warn!("[TRACE-SMP] core {cpu_id} online (CPU_ONLINE={})", CPU_ONLINE.0.load(Ordering::Relaxed));
			}
		}
	}
}

pub fn print_statistics() {
	interrupts::print_statistics();
}
