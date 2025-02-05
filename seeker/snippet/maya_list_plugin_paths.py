#date: 2025-02-05T16:47:36Z
#url: https://api.github.com/gists/4ecb17a44650764228970f0e05630186
#owner: https://api.github.com/users/BigRoy

from typing import Tuple, List
import os
import platform

# Get plugin extensions maya can load per platform
PLUGIN_EXTENSIONS_BY_PLATFORM = {
    "windows": {".mll", ".nll.dll"},
    "mac": {".bundle"},
    "linux": {".so"},
}
PLUGIN_EXTENSIONS = PLUGIN_EXTENSIONS_BY_PLATFORM[platform.system().lower()]
PLUGIN_EXTENSIONS.update({".py", ".pyc"})
PLUGIN_EXTENSIONS: Tuple[str] = tuple(PLUGIN_EXTENSIONS)


def get_plugins() -> List[str]:
    """Return all Maya plug-in filepaths found on `MAYA_PLUG_IN_PATH`.
    
    Should match behavior as the list that Maya shows in the Plug-in Manager.
    
    Returns:
        List[str]: List of full paths to individual plug-ins.
    
    """
    valid = set()
    for path in os.getenv("MAYA_PLUG_IN_PATH").split(os.pathsep):
        if path and os.path.exists(path):
            valid.add(path)
    
    plugin_paths = set()
    for folder in sorted(valid):
        for fname in os.listdir(folder):
            if not fname.endswith(plugin_extensions):
                continue
                
            path = os.path.join(folder, fname)
            if not os.path.isfile(path):
                continue
                
            plugin_paths.add(path)
     
    # Include misc plugins (anything user may have manually loaded)
    plugin_paths.update(cmds.pluginInfo(query=True, listPluginsPath=True))
            
    # Filter out .pyc matches for which a .py is found as well
    plugin_paths = {
        path for path in plugin_paths
        if not path.endswith(".pyc") and path[:-1] not in plugin_paths
    }
    
    return list(sorted(plugin_paths))
            
    
for plugin_path in get_plugins():
    print(plugin_path)