#date: 2025-01-13T16:54:49Z
#url: https://api.github.com/gists/be5e71f08d9a422da545fd0469456776
#owner: https://api.github.com/users/bendemott

from twisted.web import resource, server, proxy
from twisted.internet import reactor
from urllib.parse import urlparse

class ConditionalReverseProxy(proxy.ReverseProxyResource):
    def __init__(self, routes):
        """
        Initialize with a list of route configurations.
        routes should be a list of dicts with:
        - 'host': target host
        - 'port': target port
        - 'path': target path
        - 'condition': function that takes request and returns boolean
        """
        self.routes = routes
        # Initialize with dummy values, we'll override getChild
        super().__init__('dummy', 80, b'/')
    
    def getChild(self, path, request):
        # Convert path to bytes if it's not already
        if isinstance(path, str):
            path = path.encode('utf-8')
            
        # Find the first matching route
        for route in self.routes:
            if route['condition'](request):
                # Create a new ReverseProxyResource for this specific request
                return proxy.ReverseProxyResource(
                    route['host'],
                    route['port'],
                    b'/' + path + request.postpath
                )
        
        # Return a 404 if no route matches
        return resource.NoResource()

# Example usage
def create_proxy_server():
    # Define routing conditions
    routes = [
        {
            'host': 'api1.example.com',
            'port': 80,
            'path': b'/',
            'condition': lambda request: request.path.startswith(b'/api1')
        },
        {
            'host': 'api2.example.com',
            'port': 80,
            'path': b'/',
            'condition': lambda request: request.path.startswith(b'/api2')
        },
        {
            'host': 'default.example.com',
            'port': 80,
            'path': b'/',
            'condition': lambda request: True  # Default route
        }
    ]
    
    # Create the proxy resource
    proxy_resource = ConditionalReverseProxy(routes)
    
    # Create and start the site
    site = server.Site(proxy_resource)
    reactor.listenTCP(8080, site)
    reactor.run()

if __name__ == '__main__':
    create_proxy_server()