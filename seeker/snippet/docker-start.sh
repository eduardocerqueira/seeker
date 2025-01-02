#date: 2025-01-02T16:58:13Z
#url: https://api.github.com/gists/d197759f8b8d850df00f9de94913f163
#owner: https://api.github.com/users/yuna0x0

#!/bin/bash
docker run -it --rm --name verdaccio \
  -p 4873:4873 \
  -v ./conf:/verdaccio/conf \
  -v ./storage:/verdaccio/storage \
  -v ./plugins:/verdaccio/plugins \
  verdaccio/verdaccio
