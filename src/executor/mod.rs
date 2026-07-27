#[cfg(feature = "alloc-stats")]
mod alloc_stats;
#[cfg(feature = "net")]
pub(crate) mod device;
#[cfg(feature = "net")]
pub(crate) mod network;
pub(crate) mod task;
#[cfg(feature = "virtio-vsock")]
pub(crate) mod vsock;

use alloc::sync::Arc;
use alloc::task::Wake;
use core::pin::pin;
use core::sync::atomic::AtomicU32;
use core::task::{Context, Poll, Waker};
use core::time::Duration;

use crossbeam_utils::Backoff;
use hermit_sync::without_interrupts;

use crate::arch::kernel::core_local;
use crate::errno::Errno;
use crate::executor::task::AsyncTask;
use crate::io;
use crate::synch::futex::*;

/// WakerRegistration is derived from smoltcp's
/// implementation.
#[derive(Debug)]
pub(crate) struct WakerRegistration {
	waker: Option<Waker>,
}

impl WakerRegistration {
	pub const fn new() -> Self {
		Self { waker: None }
	}

	/// Register a waker. Overwrites the previous waker, if any.
	pub fn register(&mut self, w: &Waker) {
		match self.waker {
			// Optimization: If both the old and new Wakers wake the same task, we can simply
			// keep the old waker, skipping the clone.
			Some(ref w2) if (w2.will_wake(w)) => {}
			// In all other cases
			// - we have no waker registered
			// - we have a waker registered but it's for a different task.
			// then clone the new waker and store it
			_ => self.waker = Some(w.clone()),
		}
	}

	/// Wake the registered waker, if any.
	#[allow(dead_code)]
	pub fn wake(&mut self) {
		let Some(w) = self.waker.take() else {
			return;
		};

		w.wake();
	}
}

struct TaskNotify {
	/// Futex to wakeup a single task
	futex: AtomicU32,
}

impl TaskNotify {
	pub const fn new() -> Self {
		Self {
			futex: AtomicU32::new(0),
		}
	}

	pub fn wait(&self, timeout: Option<u64>) {
		// Wait for a futex and reset the value to zero. If the value
		// is not zero, someone already wanted to wakeup a task and stored another
		// value to the futex address. In this case, the function directly returns
		// and doesn't block.
		let _ = futex_wait_and_set(&self.futex, 0, timeout, Flags::RELATIVE, 0);
	}
}

impl Wake for TaskNotify {
	fn wake(self: Arc<Self>) {
		self.wake_by_ref();
	}

	fn wake_by_ref(self: &Arc<Self>) {
		let _ = futex_wake_or_set(&self.futex, 1, u32::MAX);
	}
}

pub(crate) fn run() {
	// INV-P10 (Spike 2): the executor must NEVER run on the exception stack
	// (it would corrupt nested-ISR state and is a thread-mode operation). Assert
	// we are NOT in an IRQ/FIQ handler. Gated on `pmr-band`; inert in default
	// build.
	#[cfg(feature = "pmr-band")]
	debug_assert!(
		!core_local::CoreLocal::get().in_irq(),
		"executor::run() called on the exception stack (INV-P10 violation)"
	);
	without_interrupts(|| {
		// FIXME: We currently have no more than 3 tasks at a time, so this is fine.
		// Ideally, we would set this value to 200, but the network task currently immediately wakes up again.
		// This would lead to the network task being polled 200 times back to back, slowing things down considerably.
		for _ in 0..3 {
			if !core_local::ex().try_tick() {
				break;
			}
		}
	});
}

/// Spawns a future on the executor.
#[cfg_attr(
	not(any(
		feature = "alloc-stats",
		feature = "shell",
		feature = "net",
		feature = "virtio-vsock"
	)),
	expect(dead_code)
)]
pub(crate) fn spawn<F>(future: F)
where
	F: Future<Output = ()> + Send + 'static,
{
	core_local::ex().spawn(AsyncTask::new(future)).detach();
}

pub fn init() {
	#[cfg(feature = "net")]
	network::init();
	#[cfg(feature = "virtio-vsock")]
	vsock::init();
	#[cfg(feature = "alloc-stats")]
	alloc_stats::init();
}

/// Blocks the current thread on `f`, running the executor when idling.
pub(crate) fn block_on<F, T>(future: F, timeout: Option<Duration>) -> io::Result<T>
where
	F: Future<Output = io::Result<T>>,
{
	let backoff = Backoff::new();
	let start = crate::arch::kernel::systemtime::now_micros();
	let task_notify = Arc::new(TaskNotify::new());
	let waker = task_notify.clone().into();
	let mut cx = Context::from_waker(&waker);
	let mut future = pin!(future);

	let timeout_ms: i64 = timeout.map(|d| d.as_millis() as i64).unwrap_or(-1);
	let mut iter: u64 = 0;

	loop {
		// check future
		let result = future.as_mut().poll(&mut cx);

		// run background all tasks, which poll also the network device
		run();

		let now = crate::arch::kernel::systemtime::now_micros();
		let elapsed_ms = (now as i128 - start as i128) as i64;
		let backoff_done = backoff.is_completed();
		debug!(
			"BLOCK_ON iter={} start={} now={} elapsed_ms={} timeout_ms={} backoff_done={} ready={}",
			iter,
			start,
			now,
			elapsed_ms,
			timeout_ms,
			backoff_done,
			matches!(result, Poll::Ready(_))
		);
		iter += 1;

		if let Poll::Ready(t) = result {
			return t;
		}

		if let Some(duration) = timeout
			&& Duration::from_micros(now - start) >= duration
		{
			warn!("BLOCK_ON timeout-elapsed -> returning Err(Time)");
			return Err(Errno::Time);
		}

		if backoff.is_completed() {
			let wakeup_time = timeout.map(|duration| u64::try_from(duration.as_micros()).unwrap());

			// switch to another task
			warn!(
				"BLOCK_ON parking via task_notify.wait(wakeup_time={:?})",
				wakeup_time
			);
			task_notify.wait(wakeup_time);
			warn!("BLOCK_ON woke from task_notify.wait");

			// restore default values
			backoff.reset();
		} else {
			backoff.snooze();
		}
	}
}
