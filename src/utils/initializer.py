import logging
import asyncio
import os
from fastapi import BackgroundTasks, FastAPI
from src.utils.startup_config import get_startup_config
from src.utils.registry import registry
from src.utils.db_manager import db_manager
from src.data_processing.build_index import build_medical_index

logger = logging.getLogger("initializer")

async def initialize_system(background_tasks: BackgroundTasks = None):
    """Initialize the system with optimizations"""
    config = get_startup_config()
    logger.info("Starting system initialization")
    
    # Check if already initialized to prevent duplicate initialization
    if registry.has("system_initialized"):
        logger.info("System already initialized, skipping initialization")
        return
    
    # Initialize essential services synchronously
    await initialize_essential_services(config)
    
    # Initialize non-essential services in the background
    if background_tasks:
        logger.info("Starting background tasks using FastAPI BackgroundTasks")
        background_tasks.add_task(initialize_background_services, config)
    else:
        logger.info("No BackgroundTasks provided, using asyncio.create_task instead")
        task = asyncio.create_task(initialize_background_services(config))
        registry.set("background_init_task", task)  # Prevent garbage collection
    
    registry.set("system_initialized", True)
    logger.info("Essential services initialized, background initialization started")

async def initialize_essential_services(config):
    """Initialize essential services that are needed immediately"""
    # Initialize database
    logger.info("Starting initialization of essential services")
    await db_manager.connect_with_retry()
    registry.set("db_manager", db_manager)
    
    # Initialize GPU FAISS if available
    try:
        import faiss
        if hasattr(faiss, 'get_num_gpus') and faiss.get_num_gpus() > 0:
            logger.info(f"Enabling GPU for FAISS with {faiss.get_num_gpus()} GPUs")
            registry.set("use_gpu_faiss", True)
        else:
            logger.info("GPU FAISS not available, using CPU version")
            registry.set("use_gpu_faiss", False)
    except Exception as e:
        logger.warning(f"Error initializing GPU FAISS: {e}")
        registry.set("use_gpu_faiss", False)
        
async def initialize_background_services(config):
    """Initialize non-essential services in the background"""
    try:
        logger.info("Starting background initialization")
        
        # Initialize retriever if needed
        logger.info("Preload_essential_only: %s", config["preload_essential_only"])
        if not config["preload_essential_only"]:
            logger.info("Initializing retriever in background")
            await initialize_retriever(config)
        
        # Initialize other background services
        # ...
        
        logger.info("Background initialization completed")
    except Exception as e:
        logger.error(f"Error in background initialization: {e}", exc_info=True)  # Add exc_info=True

async def initialize_retriever(config):
    """Initialize the document retriever with optimizations"""
    try:
        logger.info("Initializing retriever")
        documents_path = config.get("documents_path", "src/data/documents")
        index_path = config.get("index_path", "src/data/faiss_medical_index")
        
        # Check if documents directory exists and contains PDFs
        if not os.path.exists(documents_path):
            logger.error(f"Documents directory does not exist: {documents_path}")
            os.makedirs(documents_path, exist_ok=True)
            logger.info(f"Created empty documents directory: {documents_path}")
            return None
            
        pdf_files = [f for f in os.listdir(documents_path) if f.lower().endswith(".pdf")]
        if not pdf_files:
            logger.error(f"No PDF files found in documents directory: {documents_path}")
            return None
            
        logger.info(f"Found {len(pdf_files)} PDF files in {documents_path}")
        
        # Check if we need to build/rebuild the index
        rebuild_index = True
        if os.path.exists(os.path.join(index_path, "index.faiss")):
            # Check if any documents are newer than the index
            index_mtime = os.path.getmtime(os.path.join(index_path, "index.faiss"))
            doc_files = [os.path.join(documents_path, f) for f in pdf_files]
            rebuild_index = any(os.path.getmtime(path) > index_mtime for path in doc_files)
        
        if rebuild_index:
            logger.info("Documents have changed or index doesn't exist, building index...")
            # Use the bulk_process_directory function to create the index
            from src.data_processing.document_loader import MedicalDocumentProcessor
            
            # Create a document processor
            processor = MedicalDocumentProcessor(
                pdf_folder=documents_path,
                image_output_folder=os.path.join(documents_path, "extracted_images"),
                cache_dir=os.path.join(documents_path, "cache")
            )
            
            # Process all documents and build the index
            retriever = processor.bulk_process_directory(output_index_path=index_path)
            logger.info(f"Index built successfully at {index_path}")
        else:
            logger.info("Using existing index")
            from src.retrieval.document_retriever import MedicalDocumentRetriever
            
            # Check if GPU FAISS is available
            use_gpu = registry.get("use_gpu_faiss", False)
            retriever = MedicalDocumentRetriever(
                index_path=index_path, 
                use_gpu=use_gpu
            )
        
        # Store the retriever in the registry
        registry.set("retriever", retriever)
        
        logger.info("Retriever initialized successfully")
        return retriever
    except Exception as e:
        logger.error(f"Error initializing retriever: {e}")
        logger.exception(e)  # Log the full traceback
        return None
    
def init_app(app: FastAPI):
    """Initialize the application with database and retriever"""
    @app.on_event("startup")
    async def startup_event():
        # Create a BackgroundTasks instance
        background_tasks = BackgroundTasks()
        
        # Initialize the system with background tasks
        await initialize_system(background_tasks)
        
        # Execute background tasks manually since we're not in a request context
        for task in background_tasks.tasks:
            asyncio.create_task(task())
    
    @app.on_event("shutdown")
    async def shutdown_event():
        if registry.has("db_manager"):
            db_manager = registry.get("db_manager")
            await db_manager.close()
        
        logger.info("Application shutdown complete")
    
    # Add health check endpoints
    add_health_endpoints(app)


def add_health_endpoints(app: FastAPI):
    """Add health check endpoints to the FastAPI app"""
    @app.get("/health")
    async def health_check():
        """Basic health check endpoint"""
        return {"status": "healthy"}
        
    @app.get("/readiness")
    async def readiness_check():
        """Check if all components are initialized and ready"""
        ready = True
        status = {
            "database": registry.has("db_manager"),
            "retriever": registry.has("retriever"),
            "system_initialized": registry.get("system_initialized", False)
        }
        
        if not all(status.values()):
            ready = False
            
        return {
            "status": "ready" if ready else "initializing",
            "components": status
        }