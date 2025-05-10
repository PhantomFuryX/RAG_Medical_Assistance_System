import torch
import gc
from src.utils.logger import get_app_logger

logger = get_app_logger()

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

def optimize_gpu_memory():
    """Optimize GPU memory usage"""
    if torch.cuda.is_available():
        # Clear cache
        torch.cuda.empty_cache()
        # Run garbage collection
        gc.collect()
        
        # Log memory usage
        logger.info(f"GPU memory optimized. "
                   f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB, "
                   f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        
def print_gpu_info():
    """Print detailed GPU information"""
    if not torch.cuda.is_available():
        return "CUDA is not available. Using CPU."
    
    gpu_count = torch.cuda.device_count()
    info = [f"🚀 CUDA is available with {gpu_count} GPU(s). Using GPU for faster processing."]
    
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        total_memory = props.total_memory / (1024**3)  # Convert to GB
        reserved_memory = torch.cuda.memory_reserved(i) / (1024**3)
        allocated_memory = torch.cuda.memory_allocated(i) / (1024**3)
        
        info.append(f"  - GPU {i}: {props.name}")
        info.append(f"    Total memory: {total_memory:.2f} GB")
        info.append(f"    Reserved memory: {reserved_memory:.2f} GB")
        info.append(f"    Allocated memory: {allocated_memory:.2f} GB")
    
    return "\n".join(info)
