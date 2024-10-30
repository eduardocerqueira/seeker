#date: 2024-10-30T17:04:58Z
#url: https://api.github.com/gists/651ec466ac13e02a44cc6ba37c402b19
#owner: https://api.github.com/users/grizmio

from django.templatetags.static import StaticNode


def patch_static_tag(unique_per_deploy: str):
    def handle_simple_wrapper(func):
        def wrapper_func(cls, path):
            resolved_static_path = func(path)
            if resolved_static_path.endswith('.js') or resolved_static_path.endswith('.css'):
                resolved_static_path += f'?u={unique_per_deploy}'
            return resolved_static_path
        return classmethod(wrapper_func)

    StaticNode.handle_simple = handle_simple_wrapper(StaticNode.handle_simple)
   
# call it from settings.py
patch_static_tag('11111111')