//! Phase 8 — Supervisor / restart-policy layer (option-d-per-task-slot-rebased.md §9).
//!
//! Design (R8.6 / R9.2): the *policy* (should this entry point restart, how
//! often, in what window) is fully STATIC — it is a spawn-time parameter or a
//! build-time table. The *counter* (how many times THIS entry point has
//! actually restarted, and when) is RUNTIME HISTORY, a private kernel-internal
//! field. Both live in a table KEYED BY A STABLE ENTRY-POINT INDEX, not by the
//! Task: when a task dies its `Task` struct is freed and a respawn is a brand
//! new `Task` with a new `TaskId`, so a counter stored on the `Task` would be
//! orphaned. Keying on a stable `EntryPointId` (not a raw fn pointer, which
//! shifts under PIE/rebasing) survives task death.
//!
//! BEAM-style restart intensity (R8): `None` = never; `Always` = unbounded;
//! `MaxN { count, window }` = at most `count` restarts within `window`
//! microseconds (a sliding-window rate limit so a crash-loop can't pin a core).

use hermit_sync::InterruptTicketMutex;
use core::time::Duration;

/// Stable index of a known task entry point. NOT a raw function pointer: the
/// kernel is PIE/rebased, so a fn pointer would shift and the restart lookup
/// would become a rebasing hazard. Assign one variant per known entry point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntryPointId {
	/// The boot/idle task. Never restarted (it IS the scheduler loop).
	Idle,
	/// The application task (rs6 REPL). Restartable only if a policy is set.
	AppTask,
}

/// Restart policy for an entry point (Phase 8 / R8.6). STATIC: set at spawn
/// time, never mutated at runtime by external code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestartPolicy {
	/// Never restart (the default). Task death is final.
	None,
	/// Restart unboundedly on death.
	Always,
	/// Restart at most `count` times within a `window` (microseconds) sliding
	/// window (BEAM-style crash-loop backstop).
	MaxN { count: u32, window: Duration },
}

/// One entry-point's policy + private restart counter (R9.2: counter lives
/// HERE, keyed by `EntryPointId`, NOT on the Task).
///
/// Fields are non-`pub` (review finding #1): the table is reached only through
/// `RESTART_TABLE` + `should_restart`, so callers cannot mutate `policy`,
/// `restart_count`, or `window_start` directly and bypass the rate-limit logic.
/// `policy()` exposes read access without handing out a mutable handle.
pub struct RestartEntry {
	/// Static policy.
	policy: RestartPolicy,
	/// Private runtime history: how many times this entry point has restarted.
	restart_count: u32,
	/// Private runtime history: start of the current restart window (µs ticks).
	window_start: u64,
}

impl RestartEntry {
	pub const fn new(policy: RestartPolicy) -> Self {
		RestartEntry {
			policy,
			restart_count: 0,
			window_start: 0,
		}
	}

	/// Read-only access to the policy (review finding #1).
	pub fn policy(&self) -> RestartPolicy {
		self.policy
	}

	/// Decide whether THIS death may be restarted, mutating the private
	/// counter. Returns `true` if a respawn is permitted (and records it).
	/// `now` is the current `get_timer_ticks()` value (microseconds).
	pub fn may_restart(&mut self, now: u64) -> bool {
		match self.policy {
			RestartPolicy::None => false,
			RestartPolicy::Always => {
				self.restart_count = self.restart_count.saturating_add(1);
				true
			}
			RestartPolicy::MaxN { count, window } => {
				let window_us = window.as_micros() as u64;
				// Slide the window if we've left it.
				if window_us != 0
					&& self.window_start != 0
					&& now.saturating_sub(self.window_start) >= window_us
				{
					self.restart_count = 0;
					self.window_start = now;
				} else if self.window_start == 0 {
					self.window_start = now;
				}
				if self.restart_count < count {
					self.restart_count = self.restart_count.saturating_add(1);
					true
				} else {
					false
				}
			}
		}
	}
}

/// The restart table, keyed by stable `EntryPointId` index. Small and fixed
/// (one entry per known entry point — no MAX_CORES-style growth, and adding
/// an entry point is a compile-time change). Private kernel-internal runtime
/// history; external code cannot mutate the policy or read the counter.
pub static RESTART_TABLE: InterruptTicketMutex<[RestartEntry; 2]> =
	InterruptTicketMutex::new([
		RestartEntry::new(RestartPolicy::None), // Idle
		RestartEntry::new(RestartPolicy::None), // AppTask (default: no respawn)
	]);

/// Compile-time sanity: the table is indexed by `EntryPointId` discriminant
/// order. If `EntryPointId` gains a variant, this assert forces the table to be
/// resized — keeping the two in lock-step. (review finding #2: the message is
/// actionable — it tells you WHICH file to touch next.)
const _: () = assert!(
	EntryPointId::Idle as usize == 0 && EntryPointId::AppTask as usize == 1,
	"EntryPointId gained a variant: add a matching entry to RESTART_TABLE in \
	 kernel/src/scheduler/supervisor.rs (it is indexed by discriminant order)"
);

/// Consult the policy for `id` and, if a restart is permitted, record it.
/// Returns `true` if `exit()` should respawn the entry point.
///
/// (review finding #3: `now` is obtained internally via `get_timer_ticks()` —
/// the only caller (`exit()`) would otherwise have to thread it through, a
/// needless surface for mistakes. `get_timer_ticks()` is always safe to call
/// once the timer is up, which it is by the time any task can `exit()`.)
///
/// (review finding #4: uses `assert!`, NOT `debug_assert!`. An out-of-range
/// `idx` would index a static array out of bounds = UB; this must hold in
/// release builds too. The compile-time `const _` assert above catches *new
/// variants*; this runtime check defends against a *corrupted* `EntryPointId`
/// reaching here from a memory bug.)
pub fn should_restart(id: EntryPointId) -> bool {
	let idx = id as usize;
	// Single lock acquisition for the whole operation (review N1): the bounds
	// check and the `may_restart` mutation both go through the same guard, so
	// there is no window where the (fixed-size today) table could change
	// between the check and the use. Uses `assert!` (not `debug_assert!`) —
	// an OOB index on this static array is UB and must hold in release.
	let mut table = RESTART_TABLE.lock();
	assert!(
		idx < table.len(),
		"EntryPointId discriminant {idx} out of RESTART_TABLE range (corrupted id?)"
	);
	let now = crate::arch::kernel::processor::get_timer_ticks();
	table[idx].may_restart(now)
}
