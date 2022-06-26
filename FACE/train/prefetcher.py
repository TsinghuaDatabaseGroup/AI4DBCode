import torch
"""

"""
import time
class data_prefetcher():
    def __init__(self, loader):
        st = time.time()
        self.loader = iter(loader)

        self.origin_loader = iter(loader)
        # print('Generate loader took', time.time() - st)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_batch = next(self.loader)
        except StopIteration:
            self.next_batch = None
            return
        with torch.cuda.stream(self.stream):
            self.next_batch = self.next_batch.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_batch
        self.preload()
        return batch