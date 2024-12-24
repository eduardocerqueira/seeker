#date: 2024-12-24T17:09:59Z
#url: https://api.github.com/gists/aee861750bd49e768a3b496b2c67e6ec
#owner: https://api.github.com/users/Kelniit

#!/bin/bash

#

gcloud sql instances create perpus-database

#

gcloud sql users set-password root --host= "**********"=perpus-database --password=

# 

gcloud sql connect perpus-database --user --quiet