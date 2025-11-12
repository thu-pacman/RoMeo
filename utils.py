import torch

def print_gpu_memory():
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(i) / 1024**3    # GB
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"GPU {i}: Allocated {allocated:.2f}GB / Reserved {reserved:.2f}GB / Total {total:.2f}GB")

def print_peak_memory():
    for i in range(torch.cuda.device_count()):
        peak_alloc = torch.cuda.max_memory_allocated(i) 
        peak_reserved = torch.cuda.max_memory_reserved(i)
        total = torch.cuda.get_device_properties(i).total_memory
        
        peak_alloc_gb = peak_alloc / (1024 ** 3)
        peak_reserved_gb = peak_reserved / (1024 ** 3)
        total_gb = total / (1024 ** 3)
        
        print(f"GPU {i} Peak: "
              f"Allocated {peak_alloc_gb:.2f}GB / "
              f"Reserved {peak_reserved_gb:.2f}GB / "
              f"Total {total_gb:.2f}GB")
