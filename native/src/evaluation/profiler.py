import time
import tracemalloc
import torch

class TrainingProfiler:

    def __init__(self):
        self._start_time = None
        self._end_time = None
        self._peak_memory_cpu = 0
        self._peak_memory_gpu = 0

    def start(self):
        tracemalloc.start()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._start_time = time.time()

    def stop(self):
        self._end_time = time.time()
        _, self._peak_memory_cpu = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        if torch.cuda.is_available():
            self._peak_memory_gpu = torch.cuda.max_memory_allocated()

    def get_results(self) -> dict:
        total_time = (self._end_time or time.time()) - (self._start_time or 0)
        return {'total_time_s': total_time, 'peak_memory_cpu_mb': self._peak_memory_cpu / (1024 * 1024), 'peak_memory_gpu_mb': self._peak_memory_gpu / (1024 * 1024)}

def compare_profiles(results: list[dict]) -> dict:
    if not results:
        return {}
    summary = {'experiments': [], 'fastest': None, 'lowest_memory': None}
    min_time = float('inf')
    min_mem = float('inf')
    for r in results:
        name = r.get('name', 'unknown')
        t = r.get('total_time_s', float('inf'))
        m = r.get('peak_memory_cpu_mb', float('inf'))
        summary['experiments'].append({'name': name, 'time_s': t, 'memory_mb': m})
        if t < min_time:
            min_time = t
            summary['fastest'] = name
        if m < min_mem:
            min_mem = m
            summary['lowest_memory'] = name
    return summary
