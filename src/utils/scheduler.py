import asyncio
import logging
from datetime import datetime, timedelta
from src.utils.maintenance import run_maintenance_tasks

logger = logging.getLogger("scheduler")

class MaintenanceScheduler:
    def __init__(self, interval_hours=24):
        """
        Initialize the maintenance scheduler
        
        Args:
            interval_hours: Interval in hours between maintenance runs
        """
        self.interval_hours = interval_hours
        self.task = None
        
    async def start(self):
        """Start the maintenance scheduler"""
        if self.task is not None:
            logger.warning("Scheduler already running")
            return
            
        logger.info(f"Starting maintenance scheduler with {self.interval_hours} hour interval")
        self.task = asyncio.create_task(self._run_periodic())
        
    async def stop(self):
        """Stop the maintenance scheduler"""
        if self.task is not None:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
            self.task = None
            logger.info("Maintenance scheduler stopped")
    
    async def _run_periodic(self):
        """Run maintenance tasks periodically"""
        while True:
            try:
                # Run maintenance tasks
                run_maintenance_tasks()
                
                # Wait for next interval
                await asyncio.sleep(self.interval_hours * 3600)
            except asyncio.CancelledError:
                logger.info("Maintenance task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in maintenance task: {e}")
                # Wait a bit before retrying
                await asyncio.sleep(300)
