//! The Hermit kernel.
//!
//! This _library operating system_ (libOS) compiles to a static library
//! (libhermit.a) that applications can link against to create a _Unikernel_.
//!
//! The API documented here does not matter to such an application.
//! Such an application would use it's languages standard library which
//! internally calls this kernel's system call functions ([`syscalls`]).
//!
//! # Using Hermit
//!
//! To run a Rust application with Hermit, see [hermit-rs].
//!
//! To run a C or C++ application with Hermit, see [hermit-c].
//!
//! # Building the kernel manually
//!
//! You can build the kernel with default features for x86-64 like this:
//!
//! ```sh
//! cargo xtask build --arch x86_64
//! ```
//!
//! For more information, run:
//!
//! ```
//! cargo xtask build --help
//! ```
//!
//! # Features
//!
#![cfg_attr(
	not(feature = "document-features"),
	doc = "Activate the `document-features` Cargo feature to see feature docs here."
)]
#![cfg_attr(feature = "document-features", doc = document_features::document_features!())]
//!
//! [hermit-rs]: https://github.com/hermit-os/hermit-rs
//! [hermit-c]: https://github.com/hermit-os/hermit-c

#![allow(clippy::missing_safety_doc)]
#![cfg_attr(
	any(target_arch = "aarch64", target_arch = "riscv64"),
	allow(incomplete_features)
)]
#![cfg_attr(target_arch = "x86_64", feature(abi_x86_interrupt))]
#![feature(allocator_api)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(
	all(
		not(any(feature = "common-os", feature = "nostd")),
		not(target_arch = "riscv64"),
	),
	feature(linkage)
)]
#![feature(linked_list_cursors)]
#![feature(never_type)]
#![cfg_attr(
	any(target_arch = "aarch64", target_arch = "riscv64"),
	feature(specialization)
)]
#![cfg_attr(
	all(
		not(any(feature = "common-os", feature = "nostd")),
		not(target_arch = "riscv64"),
	),
	feature(thread_local)
)]
#![cfg_attr(target_os = "none", no_std)]
#![cfg_attr(target_os = "none", feature(custom_test_frameworks))]
#![cfg_attr(all(target_os = "none", test), test_runner(crate::test_runner))]
#![cfg_attr(
	all(target_os = "none", test),
	reexport_test_harness_main = "test_main"
)]
#![cfg_attr(all(target_os = "none", test), no_main)]
// FIXME: move this to `Cargo.toml` once stable
#![feature(strict_provenance_lints)]
#![warn(implicit_provenance_casts)]

// EXTERNAL CRATES
#[macro_use]
extern crate alloc;
#[macro_use]
extern crate bitflags;
#[macro_use]
extern crate log;
#[cfg(not(target_os = "none"))]
#[macro_use]
extern crate std;

#[cfg(feature = "smp")]
use core::hint::spin_loop;
#[cfg(feature = "smp")]
use core::sync::atomic::{AtomicU32, Ordering};

use self::arch::kernel;
use self::arch::kernel::core_local::{core_id, core_scheduler, CoreLocal};
use self::arch::kernel::interrupts;
use crate::alloc::string::ToString;
use crate::scheduler::{PerCoreScheduler, PerCoreSchedulerExt};

#[macro_use]
mod macros;

#[macro_use]
mod logging;

pub mod arch;
#[cfg(all(feature = "common-os", target_arch = "x86_64"))]
pub mod common_os;
pub mod config;
pub mod console;
mod drivers;
mod entropy;
mod env;
mod diagnostics;
pub mod errno;
mod executor;
pub mod fd;
pub mod fs;
mod init_buf;
mod init_cell;
pub mod io;
pub mod mm;
pub mod scheduler;
#[cfg(feature = "shell")]
mod shell;
mod synch;
pub mod syscalls;
pub mod time;
#[cfg(feature = "uhyve")]
mod uhyve;

mod built_info {
	include!(concat!(env!("OUT_DIR"), "/built.rs"));
}

hermit_entry::define_abi_tag!();

#[cfg(target_os = "none")]
hermit_entry::define_entry_version!();

#[cfg(test)]
#[cfg(target_os = "none")]
#[unsafe(no_mangle)]
extern "C" fn runtime_entry(_argc: i32, _argv: *const *const u8, _env: *const *const u8) -> ! {
	println!("Executing hermit unittests. Any arguments are dropped");
	test_main();
	core_scheduler().exit(0)
}

//https://github.com/rust-lang/rust/issues/50297#issuecomment-524180479
#[cfg(test)]
pub fn test_runner(tests: &[&dyn Fn()]) {
	println!("Running {} tests", tests.len());
	for test in tests {
		test();
	}
	core_scheduler().exit(0)
}

#[cfg(target_os = "none")]
#[test_case]
fn trivial_test() {
	println!("Test test test");
	panic!("Test called");
}

// C runtime globals from newlib (libc.a). These must be initialized before
// LIEF's C++ constructors run via `__libc_init_array` inside `runtime_entry`.
// See docs/atexit-bss-mystery.md for full analysis.
#[cfg(all(target_os = "none", not(test)))]
unsafe extern "C" {
	/// Pointer to the current `struct _atexit`. Must be NULL so
	/// `__register_exitproc` falls back to the static `__atexit0`.
	static mut __atexit: *mut core::ffi::c_void;
	/// newlib's global `struct _reent`. Must be non-zero for reentrant newlib
	/// functions (printf, malloc, atexit, etc.) to work.
	static mut __sf: core::ffi::c_void;
	/// newlib's `_impure_ptr`, read by `__getreent()` to find the current
	/// `struct _reent`. Must point at `__sf`.
	static mut _impure_ptr: *mut core::ffi::c_void;
}

/// Initialize newlib C runtime globals so LIEF's C++ static constructors
/// (which call `std::atexit`, `malloc`, etc.) work correctly.
///
/// Must be called once on core 0, before `runtime_entry` invokes
/// `__libc_init_array` which runs `.init_array` constructors.
#[cfg(all(target_os = "none", not(test)))]
unsafe fn init_c_runtime() {
	// Ensure `__atexit` is NULL. `__register_exitproc` checks this: if NULL,
	// it uses the static `__atexit0` fallback. If non-NULL (e.g. `1` due to
	// uninitialized / clobbered BSS), it dereferences the invalid pointer → fault.
	unsafe {
		__atexit = core::ptr::null_mut();
		// Point newlib's reent at the static `__sf` so reentrant functions
		// (printf, malloc, atexit, pthread_once from LIEF ctors) work.
		_impure_ptr = &raw mut __sf as *mut _;
	}
	// newlib's `__getreent()` reads the reent pointer from a TLS slot at
	// `tpidr_el0 + 0x130`. Initialize it to `&__sf` so ctor-time reent access
	// is valid before any thread-local reent is set up.
	let tpidr_el0: usize;
	unsafe {
		core::arch::asm!("mrs {0}, tpidr_el0", out(reg) tpidr_el0);
	}
	unsafe {
		*(tpidr_el0 as *mut u64).add(0x130 / 8) = &raw mut __sf as *mut _ as u64;
	}
	info!(
		"C runtime initialized: __atexit = NULL, __sf at {:p}",
		&raw const __sf
	);
}

/// Entry point of a kernel thread, which initialize the libos
#[cfg(target_os = "none")]
extern "C" fn initd(_arg: usize) {
	unsafe extern "C" {
		#[cfg(all(not(test), not(any(feature = "nostd", feature = "common-os"))))]
		fn runtime_entry(argc: i32, argv: *const *const u8, env: *const *const u8) -> !;
		#[cfg(all(not(test), any(feature = "nostd", feature = "common-os")))]
		fn main(argc: i32, argv: *const *const u8, env: *const *const u8);
	}

	// Initialize Drivers
	drivers::init();
	// The filesystem needs to be initialized before network to allow writing packet captures to a file.
	fs::init();
	executor::init();

	syscalls::init();
	#[cfg(feature = "shell")]
	shell::init();

	// Get the application arguments and environment variables.
	#[cfg(not(test))]
	let (argc, argv, environ) = syscalls::get_application_parameters();

	// give the IP thread time to initialize the network interface
	core_scheduler().reschedule();

	if cfg!(feature = "warn-prebuilt") {
		warn!("This is a prebuilt Hermit kernel.");
		warn!("For non-default device drivers and features, consider building a custom kernel.");
	}

	// Initialize the newlib C runtime globals before the application starts.
	// runtime_entry → __libc_init_array → LIEF constructors → std::atexit()
	// requires a valid C runtime. Without this, __register_exitproc faults.
	#[cfg(not(test))]
	unsafe {
		init_c_runtime();
	}

	info!("Jumping into application");

	// === B2 DIAGNOSTIC: instrumented .init_array pre-walk ===
	// Bug #3 is a branch-to-0 (PC=0, x30=0, EC=0x21) inside the C-runtime
	// static-initializer run. The `.init_array` table is statically clean (no
	// null entries) and the stack sentinel showed NO stray-write corruption,
	// so the null branch is a runtime indirect call (vtable/ifunc/callback)
	// executed from *within* one constructor. To name it without QEMU/GDB we
	// walk __init_array_start..__init_array_end ourselves, logging each ctor's
	// address immediately BEFORE calling it. The fault occurs during this walk
	// (before runtime_entry), so the LAST "[B2-CTOR]" line printed identifies
	// the offending constructor — feed that address to `objdump -d` to find the
	// null indirect call. No double-init: we crash before reaching runtime_entry.
	#[cfg(all(not(test), not(any(feature = "nostd", feature = "common-os"))))]
	unsafe {
		unsafe extern "C" {
			static __init_array_start: u8;
			static __init_array_end: u8;
		}
		let start = &raw const __init_array_start as usize;
		let end = &raw const __init_array_end as usize;
		let n = (end - start) / size_of::<usize>();
		warn!("[B2-CTOR] init_array start={start:#x} end={end:#x} count={n}");
		let table = start as *const Option<extern "C" fn()>;
		for i in 0..n {
			let slot = table.add(i);
			let fnptr = core::ptr::read(slot);
			let raw = fnptr.map_or(0usize, |f| f as usize);
			warn!("[B2-CTOR] #{i} @ {:#x} -> fn={raw:#x}", slot as usize);
			if let Some(f) = fnptr {
				// Bisect the clobber writer: scan the kernel stack for
				// the deterministic value `V` (stashed by alloc_trace on
				// the eh-pool ctor's malloc) BEFORE the ctor runs; if it
				// is absent before and present immediately after, THIS
				// ctor (or a callee) is the writer. Fixed tags (no
				// format!) to avoid perturbing the layout-sensitive fault.
				let v_before =
					syscalls::LAST_ALLOC_V.load(Ordering::Relaxed);
				if v_before != 0 {
					syscalls::scan_kstack_for_v("pre", v_before);
					warn!("[B2-VCHECK] ctor #{i} pre-scan done", i = i);
					f();
					syscalls::scan_kstack_for_v("post", v_before);
					warn!("[B2-VCHECK] ctor #{i} post-scan done", i = i);
				} else {
					f();
				}
			} else {
				warn!("[B2-CTOR] #{i} is NULL, skipping");
			}
		}
		warn!("[B2-CTOR] all {n} constructors completed without fault");
	}

	#[cfg(not(test))]
	unsafe {
		// And finally start the application.
		#[cfg(all(not(test), not(any(feature = "nostd", feature = "common-os"))))]
		runtime_entry(argc, argv, environ);
		#[cfg(all(not(test), any(feature = "nostd", feature = "common-os")))]
		main(argc, argv, environ);
	}
	#[cfg(test)]
	test_main();
}

#[cfg(feature = "smp")]
fn synch_all_cores() {
	static CORE_COUNTER: AtomicU32 = AtomicU32::new(0);

	let n = CORE_COUNTER.fetch_add(1, Ordering::SeqCst) + 1;
	let boot_cores = kernel::boot_core_count();
	warn!(
		"[TRACE-SMP] synch_all_cores: core entered n={n} boot_cores={boot_cores} (waiting for all booted cores)"
	);
	while CORE_COUNTER.load(Ordering::SeqCst) != boot_cores {
		spin_loop();
	}
	warn!("[TRACE-SMP] synch_all_cores: barrier released (all {boot_cores} booted cores present)");
}

/// Entry Point of Hermit for the Boot Processor
#[cfg(target_os = "none")]
fn boot_processor_main() -> ! {
	let sp: u64;
	unsafe {
		core::arch::asm!("mov {0}, sp", out(reg) sp, options(nostack));
	}
	warn!("[TRACE-BOOT] boot_processor_main entered, sp={:#x}", sp);
	use crate::config::USER_STACK_SIZE;

	// Initialize the kernel and hardware.
	mm::claim_initial_heap();
	hermit_sync::Lazy::force(&console::CONSOLE);
	env::init();
	// Early bootarg visibility: print before logging::init() so we can see
	// whether QEMU/loader bootargs made it into the FDT chosen.bootargs.
	// This helps debug hangs that occur before the logger is initialized.
	println!(
		"[KERNEL][EARLY] bootargs = {:?}",
		env::fdt().and_then(|f| f.chosen().bootargs())
	);
	unsafe {
		logging::init();
	}

	info!("Welcome to Hermit {}", env!("CARGO_PKG_VERSION"));
	if let Some(git_version) = built_info::GIT_VERSION {
		let dirty = if built_info::GIT_DIRTY == Some(true) {
			" (dirty)"
		} else {
			""
		};

		let opt_level = if built_info::OPT_LEVEL == "3" {
			format_args!("")
		} else {
			format_args!(" (opt-level={})", built_info::OPT_LEVEL)
		};

		info!("Git version: {git_version}{dirty}{opt_level}");
	}
	let arch = built_info::TARGET.split_once('-').unwrap().0;
	info!("Architecture: {arch}");
	info!("Enabled features: {}", built_info::FEATURES_LOWERCASE_STR);
	info!("Built on {}", built_info::BUILT_TIME_UTC);

	info!("Executable start: {:p}", elf_symbols::executable_start());
	info!("ELF header:       {:p}", elf_symbols::elf_header());
	info!("Text segment end: {:p}", elf_symbols::text_end());
	info!("Data segment end: {:p}", elf_symbols::data_end());
	info!("Executable end:   {:p}", elf_symbols::executable_end());

	if let Some(fdt) = env::fdt() {
		info!("FDT:\n{fdt:#?}");
	}

	kernel::boot_processor_init();

	#[cfg(not(target_arch = "riscv64"))]
	scheduler::add_current_core();
	warn!("[TRACE-SMP] add_current_core done, core_id={}", core_id());
	interrupts::enable();
	warn!("[TRACE-SMP] IRQs enabled, calling boot_next_processor");

	kernel::boot_next_processor();
	warn!("[TRACE-SMP] boot_next_processor returned");

	#[cfg(feature = "smp")]
	synch_all_cores();

	#[cfg(feature = "pci")]
	drivers::pci::print_information();

	// Start the initd task.
	unsafe { PerCoreScheduler::spawn(initd, 0, scheduler::task::NORMAL_PRIO, 0, USER_STACK_SIZE) };

	// Run the scheduler loop.
	PerCoreScheduler::run();
}

/// Entry Point of Hermit for an Application Processor
#[cfg(all(target_os = "none", feature = "smp"))]
fn application_processor_main() -> ! {
	warn!("[TRACE-SMP] AP entering application_processor_main, cpu_id={}", core_id());
	kernel::application_processor_init();
	warn!("[TRACE-SMP] AP application_processor_init done, cpu_id={}", core_id());
	#[cfg(not(target_arch = "riscv64"))]
	scheduler::add_current_core();
	warn!("[TRACE-SMP] AP add_current_core done, cpu_id={}", core_id());
	interrupts::enable();
	warn!("[TRACE-SMP] AP IRQs enabled, calling boot_next_processor, cpu_id={}", core_id());
	kernel::boot_next_processor();
	warn!("[TRACE-SMP] AP boot_next_processor returned, cpu_id={}", core_id());

	synch_all_cores();
	executor::init();

	// Run the scheduler loop.
	PerCoreScheduler::run();
}

#[cfg(target_os = "none")]
#[panic_handler]
fn panic(info: &core::panic::PanicInfo<'_>) -> ! {
	let core_id = core_id();
	// I4(b) (R7.8 / R9.4): if recovery code on THIS core set the per-core
	// abort-zone flag before entering the recovery / abort path, a panic here
	// must HALT IMMEDIATELY instead of calling scheduler::shutdown(1) (which
	// walks corrupted state and brings down the whole system). Per-core (not
	// a global): a global flag would wrongly force an unrelated panic on
	// another core to halt too. The flag is only ever SET post-scheduler, so
	// reading CoreLocal here is safe (it is installed by then).
	if CoreLocal::get().abort_zone.load(Ordering::Relaxed) {
		// Relaxed is sufficient here (review finding #5): the flag is only
		// ever SET under scheduler control, strictly before any panic that
		// could read it (see cleanup_tasks R9.8 note), and there is no other
		// concurrent writer on this core at panic time, so no acquire/release
		// ordering is needed — we just need the latest value this core wrote.
		panic_println!(
			"[{core_id}][PANIC] in abort zone -> hard halt (recovery path)"
		);
		loop {
			crate::arch::kernel::processor::halt();
		}
	}
	panic_println!("[{core_id}][PANIC] {info}\n");

	// DISCRIMINATOR (per-task-exception-slot-design.md R4-FU2): a userspace
	// task-1 panic reported `slice index 1610613287 (0x60000207)` — an
	// SPSR-shaped value. App code is KNOWN-GOOD, so this is KERNEL-induced
	// corruption from our per-task slot / context-switch changes. Dump the
	// current task's saved trap frame (State, 36 u64) and flag any slot
	// (esp. x-slots 5..35) holding 0x60000207, BEFORE shutdown() may switch
	// context away from the faulting task. READ-ONLY — no behavior change.
	{
		let cs = core_scheduler();
		let tid = cs.get_current_task_id();
		let lsp = cs.get_last_stack_pointer().as_u64();
		let loc = info.location().map(|l| l.to_string()).unwrap_or_default();
		error!(
			"[ABORT-DUMP] task={tid:?} frame_base={lsp:#x} panic_loc={loc} (0x60000207 = SPSR-shaped panic index)"
		);
		if lsp != 0 {
			let (hit_any, hit_x) =
				crate::diagnostics::dump_frame_magic(lsp, "panic");
			error!(
				"[ABORT-DUMP] kernel_leak? any={hit_any} x_slot={hit_x} (any slot == 0x60000207)"
			);
		}
	}

	scheduler::shutdown(1);
}
