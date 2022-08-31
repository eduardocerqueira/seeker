#date: 2022-08-31T16:50:26Z
#url: https://api.github.com/gists/b277ad36a8ae8161c99c09a0f0350332
#owner: https://api.github.com/users/LandazuriPaul

#!/bin/bash

PATTERN="*.yaml.tmpl"
OLD=".yaml.tmpl"
NEW=".tmpl.yaml"
MAXDEPTH=5

find . -maxdepth $MAXDEPTH -type f -name "$PATTERN" | sed -e 'p' -E -e "s/$OLD/$NEW/g" | xargs -n2 mv