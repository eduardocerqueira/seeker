#date: 2022-03-22T17:15:15Z
#url: https://api.github.com/gists/c8a9d8ba0f834498259f74443e6a7bd7
#owner: https://api.github.com/users/krsmedlund

class ScopedMethodRateThrottle(ScopedRateThrottle):
    """  
    If you need to set different throttles per different http method on the same view.

    # Usage

    class MyView(SomeAPIView):
        throttle_classes = [ScopedMethodRateThrottle,]
        throttle_scope = {
            "GET": "100/minute",
            "POST": "10/day",
            "DELETE": 1/day",
        }    
    
    """
    
    def allow_request(self, request, view):
        self.scope = getattr(view, self.scope_attr, dict()).get(request.method)
        if not self.scope:
            return True            
        self.rate = self.get_rate()
        self.num_requests, self.duration = self.parse_rate(self.rate)
        return super().allow_request(request, view)


    