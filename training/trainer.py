import threading
import time

import psutil


class ResourceMeasurement:
    """Sample process RSS; this observes usage, not an OS memory reservation."""
    def __enter__(self):
        self.process = psutil.Process()
        self.peak_mb = self.process.memory_info().rss / 1024**2
        self.stop = threading.Event()
        self.start = time.perf_counter()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    def _sample(self):
        while not self.stop.wait(0.02):
            self.peak_mb = max(self.peak_mb, self.process.memory_info().rss / 1024**2)

    def __exit__(self, *_):
        self.seconds = time.perf_counter() - self.start
        self.peak_mb = max(self.peak_mb, self.process.memory_info().rss / 1024**2)
        self.stop.set()
        self.thread.join()
