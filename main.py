import uvicorn
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from routes.api import router as api_router
from src.utils.settings import settings
from src.utils.gpu_utils import print_gpu_info
import logging
from src.utils.logger import get_server_logger, configure_root_logger, save_uvicorn_log_config
from src.utils.initializer import init_app
import sys
from fastapi.responses import JSONResponse
import traceback

# Configure logging
configure_root_logger()
logger = get_server_logger()
log_config_file = save_uvicorn_log_config()

origins = [
    settings.CLIENT_ORIGIN,
    settings.CLIENT_ORIGIN_ONLINE
]
origins = [origin for origin in origins if origin is not None]

app = FastAPI(
    title="Medical Assistant API",
    description="RAG-based Medical Assistant System",
    version="1.0.0"
)

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Log all unhandled exceptions"""
    logger.error(f"UNHANDLED EXCEPTION: {str(exc)}")
    logger.error(f"Request path: {request.url.path}")
    logger.error("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please check the logs."},
    )

# Print GPU info and initialize system at startup
init_app(app)

@app.on_event("startup")
async def print_gpu_info_at_startup():
    print_gpu_info()
    logger.info("GPU information printed at startup")

@app.get("/")
async def read_root():
    return {
        "message": "Welcome to the Medical Assistant Application!",
        "version": "1.0.0",
        "documentation": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    from src.utils.registry import registry
    
    try:
        # Check database connection
        if registry.has("db_manager"):
            db_manager = registry.get("db_manager")
            await db_manager.connect()
        
        # Check if retriever is initialized
        retriever_status = "initialized" if registry.has("retriever") else "not initialized"
        
        return {
            "status": "healthy",
            "database": "connected",
            "retriever": retriever_status,
            "api": "running"
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

app.include_router(api_router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    import os
    
    # Determine if we're in development or production
    is_dev = os.environ.get("ENVIRONMENT", "development").lower() == "development"
    
    # Configure reload settings
    reload_settings = {}
    if is_dev:
        reload_settings = {
            "reload": True,
            "reload_dirs": ["src"],
            "reload_delay": 2.0,
            "reload_excludes": ["logs/*", "__pycache__/*", "*.pyc", ".git/*"]
        }
    
    # Run the application
    uvicorn.run(
        "main:app", 
        log_level="info", 
        log_config=log_config_file,
        host="127.0.0.1",
        port=8000,
        **reload_settings
    )
    logger.info("Uvicorn server started")