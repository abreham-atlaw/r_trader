from concurrent.futures import ThreadPoolExecutor
import random

from .dataset import BaseDataset


class PrefetchDataset(BaseDataset):
    def __init__(self, *args, prefetch=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.prefetch = prefetch
        self.executor = ThreadPoolExecutor(max_workers=1) if prefetch else None
        self.future = None
        self._prefetch_next()

    def _prefetch_next(self):
        if self.prefetch and self.executor:
            self.future = self.executor.submit(super().__getitem__, random.randint(0, len(self) - 1))

    def __getitem__(self, idx):
        if self.prefetch and self.future:
            data = self.future.result()  # Get prefetched data
            self._prefetch_next()  # Start prefetching the next batch
            return data
        else:
            return super().__getitem__(idx)
