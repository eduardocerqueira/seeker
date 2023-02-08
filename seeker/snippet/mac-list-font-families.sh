#date: 2023-02-08T16:51:59Z
#url: https://api.github.com/gists/be86d0788e5174ca71ce0e2c05c62c2d
#owner: https://api.github.com/users/olesu

#!/usr/bin/env bash

system_profiler -json SPFontsDataType| jq -r '.SPFontsDataType[].typefaces[].family' | uniq | sort