class Registry:
    """A simple registry for storing and retrieving shared resources"""
    
    def __init__(self):
        self._resources = {}
    
    def set(self, name, resource):
        """Store a resource in the registry"""
        self._resources[name] = resource
    
    def get(self, name):
        """Get a resource from the registry"""
        return self._resources.get(name)
    
    def has(self, name):
        """Check if a resource exists in the registry"""
        return name in self._resources

# Create a singleton instance
registry = Registry()