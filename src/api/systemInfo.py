from fastapi import APIRouter
from src.utils.gpu_utils import get_gpu_info
from src.utils.cpu_utils import get_cpu_info

router = APIRouter(prefix="/System", tags=["System Info"])

@router.get("/gpu-status")
async def get_gpu_status():
    """
    Returns information about GPU availability and status
    """
    return get_gpu_info()

@router.get("/cpu-status")
async def get_gpu_status():
    """
    Returns information about GPU availability and status
    """
    return get_cpu_info()