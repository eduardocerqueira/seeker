#date: 2024-02-29T16:51:25Z
#url: https://api.github.com/gists/9f6978c9bb24b81535ea37cb2e27f1d5
#owner: https://api.github.com/users/Insighttful

def dynamic_import(path_to_module: Path, object_name: str | None = None, module_name: str | None = None):
    """
    Dynamically import a module or a specific object from a module given its path.

    If `module_name` is None, the module name is derived from the stem of `path_to_module`.
    If `object_name` is provided, return the specific object from the module.
    Otherwise, return the entire module.

    Args:
        path_to_module (Path): The file path to the module to import.
        module_name (str | None): The name to assign to the module. Defaults to None.
        object_name (str | None): The name of the object to import from the module. Defaults to None.

    Usage:
        # Example usage
        path_to_module = LIBS.joinpath("meta.py")
        
        # Load the entire module, module name derived from file name
        m = dynamic_import(path_to_module)  # from app.libs import meta as m
        
        # Load a specific object (e.g., 'meta') from the module, with dynamic module name
        meta = dynamic_import(path_to_module, "meta")  # from app.libs.meta import meta
    
    Returns:
        The requested module or object from the module.
    """

    # Set module_name based on path_to_module stem if not provided
    if module_name is None:
        module_name = path_to_module.stem

    # Ensure the directory of the module is in sys.path
    module_dir = str(path_to_module.parent)
    if module_dir not in sys.path:
        sys.path.append(module_dir)

    # Define the module spec
    spec = importlib.util.spec_from_file_location(module_name, str(path_to_module))

    # Create a new module based on the spec
    module = importlib.util.module_from_spec(spec)

    # Execute the module in its own namespace
    spec.loader.exec_module(module)

    if object_name:
        # Return the specific object from the module
        return getattr(module, object_name)
    else:
        # Return the entire module
        return module