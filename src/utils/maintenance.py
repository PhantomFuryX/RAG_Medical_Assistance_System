import logging
from src.utils.registry import registry

logger = logging.getLogger("maintenance")

def cleanup_embedding_cache(max_age_days=30):
    """
    Clean up old embedding cache files
    
    Args:
        max_age_days: Maximum age in days for cache files
        
    Returns:
        Number of files removed
    """
    embedding_cache = registry.get("embedding_cache")
    if embedding_cache:
        # Clean up cache files older than specified days
        removed_count = embedding_cache.cleanup(max_age_days=max_age_days)
        logger.info(f"Removed {removed_count} old cache files")
        return removed_count
    else:
        logger.warning("Embedding cache not found in registry")
        return 0

def run_maintenance_tasks():
    """Run all maintenance tasks"""
    logger.info("Running maintenance tasks")
    
    # Clean up embedding cache
    cleanup_embedding_cache()
    
    # Add other maintenance tasks here
    
    logger.info("Maintenance tasks completed")
