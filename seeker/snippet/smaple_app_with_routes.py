#date: 2024-07-10T17:03:29Z
#url: https://api.github.com/gists/fdd56e700496f5f49d7eb32b60a32004
#owner: https://api.github.com/users/nitori

import re


def extract_variables(endpoint: str):
    """
    Very simple route "parsing" using regex:

    The route:
    "/item/{item_id}"

    becomes:
    [
      ('string', '/item/'),
      ('variable', 'item_id'),
    ]
    """

    parts = re.split(r'\{(\w+)\}', endpoint)  # -> ["/item/", "item_id", ""]
    route_pieces = []
    for i, part in enumerate(parts):
        if i % 2 == 0:
            route_pieces.append(('string', part))
        else:
            route_pieces.append(('variable', part))

    # if the last part was a variable, it still ends with one
    # empty ('string', '') part, that we just remove.
    if route_pieces[-1][1] == '':
        route_pieces.pop()

    return route_pieces


class Route:
    """A simple route class"""

    def __init__(self, pattern: str, func):
        self.pattern = pattern
        self.func = func
        self._regex_pattern = re.compile(self._compile_as_regexp(pattern))

    def _compile_as_regexp(self, pattern: str):
        """
        A pattern like: "/item/{item_id}"
        becomes a regex like: "/item/(?P<item_id>[^/]+)"
        """
        parts = extract_variables(pattern)
        regex_parts = []
        for kind, part in parts:
            if kind == 'string':
                regex_parts.append(re.escape(part))
            else:
                regex_parts.append(rf'(?P<{part}>[^/]+)')
        return '^' + ''.join(regex_parts) + '$'

    def matches(self, request_path: str):
        return self._regex_pattern.match(request_path)


class App:
    """A simple App"""

    def __init__(self):
        self.routes = []

    def get(self, pattern: str):
        def decorator(func):
            print('This is called immediately on function definition (not when the function is called).')
            self.routes.append(Route(pattern, func))
            return func

        return decorator

    def match(self, request: str):
        """find a route that matches the request"""
        for route in self.routes:
            if m := route.matches(request):
                route.func(**m.groupdict())
                return


app = App()


@app.get("/item/{item_id}")
def get_item(item_id):
    print('hello. The item id is:', item_id)


print('Routes have been registered:')
print(app.routes)

app.match('/item/123')
