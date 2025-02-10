#date: 2025-02-10T16:44:19Z
#url: https://api.github.com/gists/da829a6592c22efe81fec84d0f99b452
#owner: https://api.github.com/users/andreiio

#!/bin/bash
# Generates terraform "moved" blocks for files based on terraform state
export OLD_MODULE="old_module_name"
export NEW_MODULE="" #LEAVE EMPTY to move to root module or enter "module.new_module." if you want to move to another module

terraform state list |
grep module.${OLD_MODULE} | cut -d. -f 3- | sed 's/\"/\\\"/g' |
xargs -I {} sh -c 'echo "moved {" && echo "  from = module.${OLD_MODULE}.{}" && echo "  to = ${NEW_MODULE}{}" && echo "}"' | sed 's/\]/\"]/g' | sed 's/\[/\["/g' | tr -d '\r' > moved-blocks.tf