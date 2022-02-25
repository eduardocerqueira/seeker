#date: 2022-02-25T16:56:12Z
#url: https://api.github.com/gists/ee976d93131dc7f9b4891d5b8e755d89
#owner: https://api.github.com/users/Rustam-Z

import importlib

# Contrived example of generating a module named as a string
full_module_name = "tables." + "tasks"

# The file gets executed upon import, as expected.
mymodule = importlib.import_module(full_module_name)

# Then you can use the module like normal
var = mymodule.COLUMNS
print(var)
