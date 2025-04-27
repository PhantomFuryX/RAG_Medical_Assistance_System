import logging
import asyncio
import os
from fastapi import BackgroundTasks, FastAPI
from src.utils.startup_config import get_startup_config
from src.utils.registry import registry
from src.utils.db_manager import db_manager
from src.data_processing.build_index import build_medical_index
from src.retrieval.embedding_manager import EmbeddingManager
from src.utils.logger import get_app_logger
from src.utils.scheduler import MaintenanceScheduler
from src.utils.maintenance import run_maintenance_tasks

logger = get_app_logger()

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
    
    # Check if db_manager already in registry
    if not registry.has("db_manager"):
        await db_manager.connect_with_retry()
        registry.set("db_manager", db_manager)
    
    # Initialize embedding manager
    if not registry.has("embedding_manager"):
        from src.retrieval.embedding_manager import EmbeddingManager
        
        # Get model name from config if available
        model_name = config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        
        # Initialize the embedding manager
        logger.info(f"Initializing embedding manager with model: {model_name}")
        embedding_manager = EmbeddingManager.get_instance(model_name)
        
        # Optionally warm up the model
        embedding_manager.warm_up()
        
        # Store in registry (this is already done in get_instance, but being explicit)
        registry.set("embedding_manager", embedding_manager)
        logger.info("Embedding manager initialized and stored in registry")
    
    # Initialize GPU FAISS if available
    try:
        if not registry.has("use_gpu_faiss"):
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
    
    logger.info("Essential services initialized")
        
async def initialize_background_services(config):
    """Initialize non-essential services in the background"""
    try:
        logger.info("Starting background initialization")
        
        # Initialize retriever if needed
        logger.info("Preload_essential_only: %s", config["preload_essential_only"])
        if not config["preload_essential_only"]:
            # Check if retriever already in registry
            if not registry.has("retriever"):
                logger.info("Initializing retriever in background")
                retriever = await initialize_retriever(config)
                if retriever:
                    registry.set("retriever", retriever)
        
        # Initialize maintenance scheduler
        if not registry.has("maintenance_scheduler"):
            # Get maintenance interval from config (default to 24 hours)
            maintenance_interval = config.get("maintenance_interval_hours", 24)
            
            scheduler = MaintenanceScheduler(interval_hours=maintenance_interval)
            await scheduler.start()
            registry.set("maintenance_scheduler", scheduler)
            logger.info(f"Maintenance scheduler started with {maintenance_interval} hour interval")
        
        logger.info("Background initialization completed")
    except Exception as e:
        logger.error(f"Error in background initialization: {e}", exc_info=True)

async def initialize_retriever(config):
    """Initialize the document retriever with optimizations"""
    try:
        logger.info("Initializing retriever")
        documents_path = config.get("documents_path", "src/data/documents")
        index_path = config.get("index_path", "src/data/faiss_medical_index")
        
        # Check if retriever already in registry
        if registry.has("retriever"):
            logger.info("Retriever already in registry, skipping initialization")
            return registry.get("retriever")
        
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
            # Use the document processor from registry if available
            if registry.has("document_processor"):
                processor = registry.get("document_processor")
                logger.info("Using document processor from registry")
            else:
                # Create a document processor
                from src.data_processing.document_loader import MedicalDocumentProcessor
                processor = MedicalDocumentProcessor(
                    pdf_folder=documents_path,
                    image_output_folder=os.path.join(documents_path, "extracted_images"),
                    cache_dir=os.path.join(documents_path, "cache")
                )
                registry.set("document_processor", processor)
            
            # Process documents and create embeddings
            documents = processor.process_all_documents()
            
            # Create and save the retriever
            from src.retrieval.document_retriever import MedicalDocumentRetriever
            
            # Check if we already have a retriever in registry
            if registry.has("document_retriever"):
                retriever = registry.get("document_retriever")
                # Update the index
                retriever.create_index(documents)
            else:
                retriever = MedicalDocumentRetriever(lazy_loading=False)
                retriever.create_index(documents)
                registry.set("document_retriever", retriever)
            
            logger.info(f"Index built and saved to {index_path}")
            return retriever
        else:
            logger.info("Loading existing index...")
            from src.retrieval.document_retriever import MedicalDocumentRetriever
            
            # Check if we already have a retriever in registry
            if registry.has("document_retriever"):
                retriever = registry.get("document_retriever")
            else:
                retriever = MedicalDocumentRetriever(lazy_loading=False)
                registry.set("document_retriever", retriever)
                
            logger.info("Retriever loaded successfully")
            return retriever
            
    except Exception as e:
        logger.error(f"Error initializing retriever: {str(e)}")
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
        
        # Stop maintenance scheduler
        if registry.has("maintenance_scheduler"):
            scheduler = registry.get("maintenance_scheduler")
            await scheduler.stop()
        
        # Run a final maintenance task to clean up
        run_maintenance_tasks()
        
        # Clean up any other resources
        if registry.has("thread_pool_executor"):
            executor = registry.get("thread_pool_executor")
            executor.shutdown(wait=False)
        
        if registry.has("process_pool_executor"):
            executor = registry.get("process_pool_executor")
            executor.shutdown(wait=False)
        
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
            "retriever": registry.has("document_retriever") or registry.has("retriever"),
            "system_initialized": registry.get("system_initialized", False),
            "embedding_model": registry.has("embedding_manager"),
            "faiss_index": registry.has("faiss_index")
        }
        
        if not all(status.values()):
            ready = False
            
        return {
            "status": "ready" if ready else "initializing",
            "components": status
        }
