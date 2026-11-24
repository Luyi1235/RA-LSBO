"""Console progress for queued, active and settled X-57 evaluations."""

from collections import Counter
from contextlib import contextmanager
from threading import Event, Lock, Thread
from time import monotonic


class CampaignProgress:
    def __init__(self, total, shared_total, workers, interval=15):
        self.total = total
        self.shared_total = shared_total
        self.workers = workers
        self.interval = interval
        self.started = monotonic()
        self.latest = None
        self.phase_name = "prepare"
        self.counts = Counter()
        self.seen = set()
        self.lock = Lock()
        self.stop = Event()
        self.thread = Thread(target=self._heartbeat, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()

    def _show(self, detail=""):
        elapsed = monotonic() - self.started
        settled = len(self.seen)
        filled = 20 * settled // self.total
        bar = "#" * filled + "-" * (20 - filled)
        last = "none" if self.latest is None else f"{monotonic() - self.latest:.0f}s ago"
        message = (
            f"[progress] {self.phase_name} [{bar}] {settled}/{self.total} ({settled/self.total:.1%})"
            f" | shared {self.counts['shared']}/{self.shared_total}"
            f" | new {self.counts['new']} | replay {self.counts['replay']}"
            f" | evaluating {self.counts['evaluating']}/{self.workers}"
            f" | selecting {self.counts['selecting']}"
            f" | elapsed {elapsed:.0f}s | last settlement {last}"
        )
        print(f"{message} | {detail}" if detail else message, flush=True)

    def _heartbeat(self):
        while not self.stop.wait(self.interval):
            with self.lock:
                self._show("heartbeat")

    def phase(self, name):
        with self.lock:
            self.phase_name = name
            self._show()

    @contextmanager
    def activity(self, name):
        with self.lock:
            self.counts[name] += 1
        try:
            yield
        finally:
            with self.lock:
                self.counts[name] -= 1

    def settle(self, job, feedback, *, shared, replayed):
        with self.lock:
            if job in self.seen:
                return
            self.seen.add(job)
            self.counts["replay" if replayed else "new"] += 1
            self.counts["shared"] += shared
            self.latest = monotonic()
            self._show(f"{job} | {feedback['outcome']}{' (replay)' if replayed else ''}")
