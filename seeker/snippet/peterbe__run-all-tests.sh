#date: 2023-04-07T16:53:53Z
#url: https://api.github.com/gists/a68d4104071d8c6f0586287e49dd366b
#owner: https://api.github.com/users/peterbe

#!/bin/bash

# set -e

Time jest tests/unit tests/meta src/events tests/graphql
Time jest src/search/tests
Time jest tests/routing
Time jest tests/content
Time jest tests/rendering/
ENABLED_LANGUAGES=all Time jest tests/translations
ROOT=tests/fixtures Time jest tests/rendering-fixtures
# last because it's so slow
Time jest tests/linting
