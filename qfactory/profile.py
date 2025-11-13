import os
import time
import numpy
import torch
import logging

from triton.testing import do_bench

logger = logging.getLogger(__name__)

if torch.cuda.is_available():
    # 80 MB should be larger than L2 Cache
    _FLUSH_X = torch.empty(int(80 * (1024 ** 2)), dtype=torch.int8, device='cuda')

def _flush_cache():
    _FLUSH_X.zero_()

def profile_latency(f, warmup=200, niter=100, clear_cache=True):
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(niter)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(niter)]

    def call(iters, record=False):
        for _ in range(iters):
            if clear_cache:
                _flush_cache()
            if record:
                start_events[_].record()
            f()
            if record:
                end_events[_].record()

    torch.cuda.synchronize()
    call(warmup)
    torch.cuda.synchronize()
    torch.cuda._sleep(1_000_000_000)
    call(warmup)
    call(niter, record=True)
    torch.cuda.synchronize()

    # Make sure there is enough to "cool down" the GPU in between benchmarks to avoid throttling for later runs when
    # we execute many benchmarks consecutively
    time.sleep(1.0)

    times = [s.elapsed_time(e) * 1000 for s, e in zip(start_events, end_events)] # unit: us

    avg_time = sum(times) / len(times)

    max_rel_error = numpy.abs(numpy.array(times) - avg_time) / avg_time

    if max_rel_error.max() >= 0.05:
        logger.warning(f"Max relative error {max_rel_error.max()} exceeds 5%: min_time={min(times)}, max_time={max(times)}, avg_time={avg_time}")

    return avg_time

if os.getenv('QFACTORY_FAST_PROFILE'):
    logger.warning("Using Triton Profiler")
    profile_latency = lambda f: do_bench(f) * 1e3
