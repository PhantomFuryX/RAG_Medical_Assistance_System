import torch

def get_gpu_info():
    """
    Returns information about CUDA availability and GPU devices
    """
    if not torch.cuda.is_available():
        return {
            "cuda_available": False,
            "message": "CUDA is not available. Using CPU for processing."
        }
    
    # Get information about available GPUs
    gpu_count = torch.cuda.device_count()
    gpu_info = []
    
    for i in range(gpu_count):
        gpu_info.append({
            "index": i,
            "name": torch.cuda.get_device_name(i),
            "memory_total": torch.cuda.get_device_properties(i).total_memory / (1024**3),  # Convert to GB
            "memory_reserved": torch.cuda.memory_reserved(i) / (1024**3),  # Convert to GB
            "memory_allocated": torch.cuda.memory_allocated(i) / (1024**3)  # Convert to GB
        })
    
    return {
        "cuda_available": True,
        "gpu_count": gpu_count,
        "devices": gpu_info,
        "message": f"CUDA is available with {gpu_count} GPU(s). Using GPU for faster processing."
    }

def print_gpu_info():
    """
    Prints information about CUDA availability and GPU devices
    """
    info = get_gpu_info()
    
    if info["cuda_available"]:
        print(f"🚀 {info['message']}")
        for device in info["devices"]:
            print(f"  - GPU {device['index']}: {device['name']}")
            print(f"    Total memory: {device['memory_total']:.2f} GB")
            print(f"    Reserved memory: {device['memory_reserved']:.2f} GB")
            print(f"    Allocated memory: {device['memory_allocated']:.2f} GB")
    else:
        print(f"⚠️ {info['message']}")
