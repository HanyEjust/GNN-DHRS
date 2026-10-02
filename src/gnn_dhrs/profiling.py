import time, tracemalloc, torch

def profile_call(fn,*args,**kwargs):
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    tracemalloc.start(); t=time.perf_counter(); out=fn(*args,**kwargs)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    sec=time.perf_counter()-t; _,cpu_peak=tracemalloc.get_traced_memory(); tracemalloc.stop()
    gpu_peak=torch.cuda.max_memory_allocated()/1024**2 if torch.cuda.is_available() else 0.0
    return out,{'seconds':sec,'cpu_peak_mb':cpu_peak/1024**2,'gpu_peak_mb':gpu_peak}

def parameter_count(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)
