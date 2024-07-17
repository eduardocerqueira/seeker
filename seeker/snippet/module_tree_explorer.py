#date: 2024-07-17T16:52:13Z
#url: https://api.github.com/gists/28b193369efeee28f8f8f36d7202e03b
#owner: https://api.github.com/users/Fasteroid

import sys
import pkgutil
import importlib

def get_all_modules():
    all_modules = set(sys.modules.keys())
    
    # Add all discoverable modules
    for module in pkgutil.iter_modules():
        all_modules.add(module.name)
    
    return sorted(all_modules)

def print_module_tree(module_name, out, prefix="", last=True, visited=None):
    if visited is None:
        visited = set()
    
    if module_name in visited:
        return
    visited.add(module_name)

    def print_and_write(text):
        print(text)
        out.write(text + "\n")

    connector = "├───┬───"
    print_and_write(f"{prefix}{connector}{module_name}")

    new_prefix = prefix + "│   "

    try:
        # Use importlib.util.find_spec to check if the module exists without importing it
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            return

        # Get the module's attributes without importing it
        loader = spec.loader
        if hasattr(loader, 'get_code'):
            code = loader.get_code(module_name)
            if code:
                items = [name for name in code.co_names if not name.startswith('_')]
            else:
                items = []
        else:
            items = []

        for i, name in enumerate(items):
            is_last = (i == len(items) - 1)
            child_connector = "└─── " if is_last else "├─── "
            print_and_write(f"{new_prefix}{child_connector}{name}: attribute")

        # Check for potential submodules
        for _, submodule_name, is_pkg in pkgutil.iter_modules([spec.submodule_search_locations[0]] if spec.submodule_search_locations else []):
            full_submodule_name = f"{module_name}.{submodule_name}"
            is_last = (submodule_name == items[-1] if items else True)
            print_module_tree(full_submodule_name, out, new_prefix, is_last, visited)

    except Exception as e:
        print_and_write(f"{new_prefix}└─── Error: {str(e)}")

all_modules = get_all_modules()

# Open a file for writing with UTF-8 encoding
with open('module_tree.txt', 'w', encoding='utf-8') as f:
    # Write tree structure of each module to file
    for i, module_name in enumerate(all_modules):
        is_last = (i == len(all_modules) - 1)
        print_module_tree(module_name, f, "", is_last)
        f.write("\n")  # Add a blank line between top-level modules for readability

print("done")