use alloc::collections::BTreeMap;

use hermit_sync::InterruptTicketMutex;

use crate::arch::kernel::core_local::*;
use crate::arch::kernel::processor::{get_frequency, get_timestamp};
use crate::config::USER_STACK_SIZE;
use crate::errno::Errno;
use crate::scheduler::PerCoreSchedulerExt;
use crate::scheduler::task::{Priority, TaskHandle, TaskId};
use crate::time::timespec;
use crate::{arch, scheduler};

#[cfg(feature = "newlib")]
pub type SignalHandler = extern "C" fn(i32);
pub type Tid = i32;

#[hermit_macro::system]
#[unsafe(no_mangle)]
pub extern "C" fn sys_getpid() -> Tid {
	0
}

#[cfg(feature = "newlib")]
#[hermit_macro::system(errno)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sys_getprio(id: *const Tid) -> i32 {
	let task = core_scheduler().get_current_task_handle();

	if id.is_null() || unsafe { *id } == task.get_id().into() {
		i32::from(task.get_priority().into())
	} else {
		-i32::from(Errno::Inval)
	}
}

#[cfg(feature = "newlib")]
#[hermit_macro::system(errno)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sys_setprio(_id: *const Tid, _prio: i32) -> i32 {
	-i32::from(Errno::Nosys)
}

fn exit(arg: i32) -> ! {
	debug!("Exit program with error code {arg}!");
	super::shutdown(arg)
}

#[hermit_macro::system]
#[unsafe(no_mangle)]
pub extern "C" fn sys_exit(status: i32) -> ! {
	exit(status)
}

#[hermit_macro::system]
#[unsafe(no_mangle)]
pub extern "C" fn sys_thread_exit(status: i32) -> ! {
	debug!("Exit thread with error code {status}!");
	core_scheduler().exit(status)
}

#[hermit_macro::system]
#[unsafe(no_mangle)]
pub extern "C" fn sys_abort() -> ! {
	// DISCRIMINATOR (per-task-exception-slot-design.md R4-FU2): userspace
	// task-1 panic `slice index 1610613287 (0x60000207)` — SPSR-shaped, KNOWN
	// app code => kernel-induced corruption. Dump current task's saved State
	// (36 u64) and flag any slot (esp. x-slots 5..35) == 0x60000207. Covers
	// the app-panic -> sys_abort syscall route. READ-ONLY.
	{
		let cs = core_scheduler();
		let tid = cs.get_current_task_id();
		let lsp = cs.get_last_stack_pointer().as_u64();
		error!(
			"[ABORT-DUMP] task={tid:?} frame_base={lsp:#x} (0x60000207 = SPSR-shaped panic index)"
		);
		if lsp != 0 {
			let slot = lsp as *const u64;
			let mut hit_any = false;
			let mut hit_x = false;
			for i in 0..36u64 {
				let v = unsafe { core::ptr::addr_of!(*slot.add(i as usize)).read_volatile() };
				let is_x = i >= 5 && i <= 35;
				if v == 0x60000207 {
					hit_any = true;
					if is_x {
						hit_x = true;
					}
					error!(
						"[ABORT-DUMP] slot[{i}] @+{:#x} = 0x60000207  <<< MATCH (is_x={is_x})",
						8 * i
					);
				} else if i == 2 {
					error!("[ABORT-DUMP] slot[{i}] @+{:#x} = {:#x}  (spsr)", 8 * i, v);
				} else if i == 1 {
					error!("[ABORT-DUMP] slot[{i}] @+{:#x} = {:#x}  (elr)", 8 * i, v);
				}
			}
			error!(
				"[ABORT-DUMP] kernel_leak? any={hit_any} x_slot={hit_x} (any slot == 0x60000207)"
			);
		}
	}
	exit(-1)
}

/// Phase 7 Stage 0 live-task injection harness (option-d-per-task-slot-rebased.md §6.2).
/// A dedicated, NUMERIC test-fault syscall (`SYS_TEST_FAULT = 0xBEEF`) that drives a
/// genuine task KILL through the real recovery path: `scheduler::abort()` -> `exit(-1)`
/// -> the task becomes `Finished` -> `reschedule()` keeps the rest of the system alive
/// (I5 liveness). This is the controlled, demonstrable kill+resume the harness needs.
///
/// NOTE on mechanism: a raw `udf #0` from userspace does NOT reach `panic!()` — `do_sync`
/// spins forever on an "Unknown exception" (interrupts.rs). So the harness uses this
/// explicit syscall (the same kill path `sys_abort` takes) rather than a synthetic fault.
/// The rs6 REPL `/fault` command invokes it. Returns `!` (it never returns).
///
/// (review finding #16: gated behind `debug_assertions`. In a release build this symbol
/// does not exist, so a production userspace task cannot invoke a self-kill test hook;
/// the dev kernel + dev-built rs6 both keep it for the harness. No cross-crate feature
/// wiring needed — both sides are governed by the same cargo profile.)
#[cfg(debug_assertions)]
#[unsafe(no_mangle)]
pub extern "C" fn sys_test_fault() -> ! {
	// INVARIANT: only a real, running task can call a syscall, so
	// core_scheduler() is always installed here — no pre-scheduler guard needed.
	let tid = core_scheduler().get_current_task_id();
	info!("[STAGE0-FAULT] SYS_TEST_FAULT invoked by task {tid:?} -> scheduler::abort() (kill+resume harness)");
	crate::scheduler::abort()
}

pub(super) fn usleep(usecs: u64) {
	if usecs >= 10_000 {
		// Enough time to set a wakeup timer and block the current task.
		debug!("sys_usleep blocking the task for {usecs} microseconds");
		let wakeup_time = arch::kernel::processor::get_timer_ticks() + usecs;
		let core_scheduler = core_scheduler();
		core_scheduler.block_current_task(Some(wakeup_time));

		// Switch to the next task.
		core_scheduler.reschedule();
	} else if usecs > 0 {
		// Not enough time to set a wakeup timer, so just do busy-waiting.
		let end = get_timestamp() + u64::from(get_frequency()) * usecs;
		while get_timestamp() < end {
			core_scheduler().reschedule();
		}
	}
}

#[hermit_macro::system]
#[unsafe(no_mangle)]
pub extern "C" fn sys_msleep(ms: u32) {
	usleep(u64::from(ms) * 1000);
}

#[hermit_macro::system(errno)]
#[unsafe(no_mangle)]
pub extern "C" fn sys_usleep(usecs: u64) {
	usleep(usecs);
}

#[hermit_macro::system(errno)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sys_nanosleep(rqtp: *const timespec, _rmtp: *mut timespec) -> i32 {
	assert!(
		!rqtp.is_null(),
		"sys_nanosleep called with a zero rqtp parameter"
	);
	let requested_time = unsafe { &*rqtp };
	if requested_time.tv_sec < 0 || requested_time.tv_nsec > 999_999_999 {
		debug!("sys_nanosleep called with an invalid requested time, returning -EINVAL");
		return -i32::from(Errno::Inval);
	}

	let microseconds =
		(requested_time.tv_sec as u64) * 1_000_000 + (requested_time.tv_nsec as u64) / 1_000;
	usleep(microseconds);

	0
}

/// Creates a new thread based on the configuration of the current thread.
#[cfg(feature = "newlib")]
#[hermit_macro::system(errno)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sys_clone(id: *mut Tid, func: extern "C" fn(usize), arg: usize) -> i32 {
	let task_id = core_scheduler().clone(func, arg);

	if !id.is_null() {
		unsafe {
			*id = task_id.into();
		}
	}

	0
}

#[hermit_macro::system(errno)]
#[unsafe(no_mangle)]
pub extern "C" fn sys_yield() {
	core_scheduler().reschedule();
}

#[cfg(feature = "newlib")]
#[hermit_macro::system(errno)]
#[unsafe(no_mangle)]
pub extern "C" fn sys_kill(dest: Tid, signum: i32) -> i32 {
	debug!("sys_kill is unimplemented, returning -ENOSYS for killing {dest} with signal {signum}");
	-i32::from(Errno::Nosys)
}

#[cfg(feature = "newlib")]
#[hermit_macro::system(errno)]
#[unsafe(no_mangle)]
pub extern "C" fn sys_signal(_handler: SignalHandler) -> i32 {
	debug!("sys_signal is unimplemented");
	0
}

#[hermit_macro::system]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sys_spawn2(
	func: unsafe extern "C" fn(usize),
	arg: usize,
	prio: u8,
	stack_size: usize,
	selector: isize,
) -> Tid {
	unsafe { scheduler::spawn(func, arg, Priority::from(prio), stack_size, selector).into() }
}

#[hermit_macro::system]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sys_spawn(
	id: *mut Tid,
	func: unsafe extern "C" fn(usize),
	arg: usize,
	prio: u8,
	selector: isize,
) -> i32 {
	let new_id = unsafe {
		scheduler::spawn(func, arg, Priority::from(prio), USER_STACK_SIZE, selector).into()
	};

	if !id.is_null() {
		unsafe {
			*id = new_id;
		}
	}

	0
}

#[hermit_macro::system]
#[unsafe(no_mangle)]
pub extern "C" fn sys_join(id: Tid) -> i32 {
	match scheduler::join(TaskId::from(id)) {
		Ok(()) => 0,
		_ => -i32::from(Errno::Inval),
	}
}

/// Mapping between blocked tasks and their TaskHandle
static BLOCKED_TASKS: InterruptTicketMutex<BTreeMap<TaskId, TaskHandle>> =
	InterruptTicketMutex::new(BTreeMap::new());

fn block_current_task(timeout: Option<u64>) {
	let wakeup_time = timeout.map(|t| arch::kernel::processor::get_timer_ticks() + t * 1000);
	let core_scheduler = core_scheduler();
	let handle = core_scheduler.get_current_task_handle();
	let tid = core_scheduler.get_current_task_id();

	BLOCKED_TASKS.lock().insert(tid, handle);
	core_scheduler.block_current_task(wakeup_time);
}

/// Set the current task state to `blocked`
#[hermit_macro::system]
#[unsafe(no_mangle)]
pub extern "C" fn sys_block_current_task() {
	block_current_task(None);
}

/// Set the current task state to `blocked`
#[hermit_macro::system]
#[unsafe(no_mangle)]
pub extern "C" fn sys_block_current_task_with_timeout(timeout: u64) {
	block_current_task(Some(timeout));
}

/// Wake up the task with the identifier `id`
#[hermit_macro::system]
#[unsafe(no_mangle)]
pub extern "C" fn sys_wakeup_task(id: Tid) {
	let task_id = TaskId::from(id);

	let Some(handle) = BLOCKED_TASKS.lock().remove(&task_id) else {
		return;
	};

	core_scheduler().custom_wakeup(handle);
}

/// Determine the priority of the current thread
#[hermit_macro::system(errno)]
#[unsafe(no_mangle)]
pub extern "C" fn sys_get_priority() -> u8 {
	core_scheduler().get_current_task_prio().into()
}

/// Set priority of the thread with the identifier `id`
#[hermit_macro::system(errno)]
#[unsafe(no_mangle)]
pub extern "C" fn sys_set_priority(id: Tid, prio: u8) {
	if prio > 0 {
		core_scheduler()
			.set_priority(TaskId::from(id), Priority::from(prio))
			.expect("Unable to set priority");
	} else {
		panic!("Invalid priority {prio}");
	}
}

/// Set priority of the current thread
#[hermit_macro::system]
#[unsafe(no_mangle)]
pub extern "C" fn sys_set_current_task_priority(prio: u8) {
	if prio > 0 {
		core_scheduler().set_current_task_priority(Priority::from(prio));
	} else {
		panic!("Invalid priority {prio}");
	}
}
