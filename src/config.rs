pub(crate) const KERNEL_STACK_SIZE: usize = 0x8000;

pub const DEFAULT_STACK_SIZE: usize = 0x0002_0000;

/// Byte size of the arch-specific `State` frame (36 u64s = 288 B on aarch64).
/// This file cannot import `State` (platform-independent), so the literal is
/// defined here and cross-checked at compile time against `size_of::<State>()`
/// in `slot_pool.rs` (`STATE_SIZE == ARCH_STATE_SIZE`). If the State layout
/// changes, BOTH sites must be updated — but only ONE needs to fail to catch it.
pub(crate) const ARCH_STATE_SIZE: usize = 288;

/// Per-task exception scratch-slot size (per-task-exception-slot-design.md §3.2).
/// MUST be a multiple of the page size (BasePageSize = 0x1000): the State frame
/// (288 B) lives at the slot body's TAIL, and protect_stack_guards() unmaps the
/// per-slot GUARD page. If the body were not page-aligned, the frame's page and
/// the guard's page would overlap and unmapping the guard would zap the frame.
///
/// SIZING (R4, 2026-07-24): the slot is a full EXCEPTION STACK, not merely a
/// State frame holder. On a sync exception HW sets SP = SP_EL1 = slot TOP;
/// trap_entry pushes State (0x120) and then `do_sync`/`do_irq`/... run their
/// ENTIRE call chain on this stack (logging + serial mutex + formatting +
/// core_scheduler + a possible nested switch). A debug-build `do_sync` prologue
/// alone reserves ~0x25a0 B (stack-clash probed), which blew the old 0x2000
/// slot straight through the lower guard (data abort at slot_base-0x2010).
/// Size it like the old shared per-core E / HERMIT_EARLY_EXCEPTION_STACK_POOL:
/// 64 KiB.
pub const EXCEPTION_SLOT_SIZE: usize = 0x10000;
// Invariant guards (R3/R4): these catch (a) the page-aliasing bug (body page ==
// guard page) and (b) an UNDERSIZED slot that cannot hold the exception-handler
// call chain. (b) is the real trap: a `>= State + marker` bound is far too weak
// — `do_sync`'s own debug frame is ~0x25a0 B. Require a real exception-stack
// size so the handler + a nested switch fit. These fail at COMPILE TIME.
const _: () = assert!(
	EXCEPTION_SLOT_SIZE % 0x1000 == 0,
	"EXCEPTION_SLOT_SIZE must be a multiple of the page size (0x1000) so the frame page and guard page are disjoint"
);
const _: () = assert!(
	EXCEPTION_SLOT_SIZE >= 0x8000,
	"EXCEPTION_SLOT_SIZE must be a real exception-stack size (>= 32 KiB): it holds State PLUS the full handler call chain (do_sync's debug frame alone is ~0x25a0 B), not just a State frame"
);
/// Number of scratch slots per core. 3 = running + blocked + dispatch slack.
pub const SLOTS_PER_CORE: usize = 3;
const _: () = assert!(
	SLOTS_PER_CORE >= 2,
	"SLOTS_PER_CORE must be >= 2 (running + at least one blocked for eviction)"
);
/// Guard tail per slot element (mirrors .exception_stacks: one unmapped 4 KiB
/// page after each slot so an overflow faults instead of corrupting the next).
pub(crate) const EXCEPTION_SLOT_GUARD: usize = 4096;
const _: () = assert!(
	EXCEPTION_SLOT_GUARD >= 0x1000,
	"EXCEPTION_SLOT_GUARD must be >= one page (0x1000) for protect_stack_guards"
);

pub(crate) const USER_STACK_SIZE: usize = 0x0010_0000;

#[cfg(feature = "virtio")]
#[allow(dead_code)]
pub(crate) const VIRTIO_MAX_QUEUE_SIZE: u16 = if cfg!(feature = "pci") { 2048 } else { 1024 };

/// Default keep alive interval in milliseconds
#[cfg(feature = "tcp")]
pub(crate) const DEFAULT_KEEP_ALIVE_INTERVAL: u64 = 75000;

#[cfg(feature = "virtio-vsock")]
pub(crate) const VSOCK_PACKET_SIZE: u32 = 8192;

#[cfg(feature = "virtio-console")]
pub(crate) const CONSOLE_PACKET_SIZE: u32 = 8192;
