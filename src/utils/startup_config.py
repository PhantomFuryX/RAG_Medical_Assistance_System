import os

# Startup optimization settings
STARTUP_CONFIG = {
    # General settings
    "lazy_loading": True,
    "use_cache": True,
    "preload_essential_only": False,
    
    # Document processing
    "parallel_processing": True,
    "max_workers": os.cpu_count() - 1 if os.cpu_count() > 1 else 1,
    "chunk_size": 1000,
    "chunk_overlap": 200,
    
    # Model settings
    "use_quantized_models": True,
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "device": "cuda" if os.environ.get("USE_GPU", "true").lower() == "true" else "cpu",
    
    # Cache settings
    "cache_dir": "src/data/cache",
    "embeddings_cache_dir": "src/data/cache/embeddings",
    "documents_cache_dir": "src/data/cache/documents",
    "index_cache_dir": "src/data/cache/index",
    
    # Paths
    "documents_path": os.environ.get("DOCUMENTS_PATH", "src/data/medical_books"),
    "embeddings_path": os.environ.get("EMBEDDINGS_PATH", "src/data/embeddings"),
    "index_path": os.environ.get("INDEX_PATH", "src/data/faiss_medical_index"),
}

def get_startup_config():
    """Get the startup configuration with environment variable overrides"""
    config = STARTUP_CONFIG.copy()
    
    # Override with environment variables
    for key in config:
        env_key = f"STARTUP_{key.upper()}"
        if env_key in os.environ:
            # Convert the value to the appropriate type
            env_value = os.environ[env_key]
            if isinstance(config[key], bool):
                config[key] = env_value.lower() == "true"
            elif isinstance(config[key], int):
                config[key] = int(env_value)
            else:
                config[key] = env_value
    
    return config
