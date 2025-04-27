"""
Registry module for storing and retrieving global objects
"""
from typing import Any, Dict, Optional
import threading

class Registry:
    """
    A thread-safe singleton registry for storing and retrieving global objects
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(Registry, cls).__new__(cls)
                cls._instance._registry = {}
        return cls._instance
    
    def set(self, key: str, value: Any) -> None:
        """Set a value in the registry"""
        with self._lock:
            self._registry[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the registry with optional default"""
        with self._lock:
            return self._registry.get(key, default)
    
    def has(self, key: str) -> bool:
        """Check if a key exists in the registry"""
        with self._lock:
            return key in self._registry
    
    def remove(self, key: str) -> None:
        """Remove a key from the registry"""
        with self._lock:
            if key in self._registry:
                del self._registry[key]
    
    def clear(self) -> None:
        """Clear the entire registry"""
        with self._lock:
            self._registry.clear()
    
    def get_all(self) -> Dict[str, Any]:
        """Get a copy of all registry items"""
        with self._lock:
            return self._registry.copy()

# Create a singleton instance
registry = Registry()