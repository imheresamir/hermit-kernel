//! Shared frame-diagnostic helper (review B7/C7).
//!
//! The `ABORT-DUMP` magic-value scan was duplicated (verbatim) in both
//! `scheduler::exit()` and `lib.rs`'s panic handler. That duplication is a
//! maintenance hazard: any change to the scan must be made in two places.
//! This module centralizes it.
//!
//! WHY THE SCAN EXISTS (per-task-exception-slot-design.md R4-FU2): a userspace
//! task-1 panic once reported `slice index 1610613287 (0x60000207)` — an
//! SPSR-shaped value. App code is KNOWN-GOOD, so that was KERNEL-induced
//! corruption from the per-task slot / context-switch path. The scan dumps the
//! saved trap frame (State, 36 u64 words) and flags any word equal to the
//! magic `0x60000207`, so a recurrence is immediately visible. READ-ONLY — it
//! never changes behavior, only prints.

/// The SPSR-shaped magic value that signalled slot-frame corruption during the
/// bring-up investigation.
pub const MAGIC_SLOT_CORRUPTION: u64 = 0x60000207;

/// Number of `u64` words in a saved `State` trap frame (ARCH_STATE_SIZE / 8).
pub const STATE_WORDS: usize = crate::config::ARCH_STATE_SIZE / size_of::<u64>();

/// Scan the saved trap frame at `frame_base` and report any word equal to
/// [`MAGIC_SLOT_CORRUPTION`]. `label` is printed in each hit line so callers
/// from `exit()` vs the panic handler remain distinguishable. Returns
/// `(hit_any, hit_x_slot)` where `hit_x_slot` is true if a match landed in the
/// x-register region (word indices 5..35).
///
/// # Safety
///
/// `frame_base` must be the address of a fully-mapped `State` frame (36 u64s,
/// naturally aligned). Callers pass `last_stack_pointer`, which is either 0
/// (skipped by the caller before calling) or a valid slot/kernel-stack frame
/// address. The read uses `addr_of!().read_volatile()` so a partially-corrupt
/// frame still reads without trapping on a bad pointer deref.
pub fn dump_frame_magic(frame_base: u64, label: &str) -> (bool, bool) {
	let slot = frame_base as *const u64;
	let mut hit_any = false;
	let mut hit_x = false;
	for i in 0..STATE_WORDS as u64 {
		// SAFETY: `frame_base` is a mapped, aligned State frame (caller
		// guarantees non-zero and valid). `read_volatile` avoids any
		// optimizer assumption about the (possibly corrupted) contents and
		// tolerates reading a word even if the frame is mid-corruption.
		let v = unsafe { core::ptr::addr_of!(*slot.add(i as usize)).read_volatile() };
		// State layout: spsel@0, elr@8, spsr@16, sp_el0@24, x0@40..x30@280
		// => x-register slots are u64 indices 5..35.
		let is_x = i >= 5 && i <= 35;
		if v == MAGIC_SLOT_CORRUPTION {
			hit_any = true;
			if is_x {
				hit_x = true;
			}
			error!(
				"[ABORT-DUMP] {label} slot[{i}] @+{:#x} = 0x60000207  <<< MATCH (is_x={is_x})",
				8 * i
			);
		} else if i == 2 {
			error!("[ABORT-DUMP] {label} slot[{i}] @+{:#x} = {:#x}  (spsr)", 8 * i, v);
		} else if i == 1 {
			error!("[ABORT-DUMP] {label} slot[{i}] @+{:#x} = {:#x}  (elr)", 8 * i, v);
		}
	}
	(hit_any, hit_x)
}
