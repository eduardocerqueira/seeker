#date: 2022-11-08T17:18:47Z
#url: https://api.github.com/gists/5d5fdc235672b42d49e6907634064dfa
#owner: https://api.github.com/users/leaver2000

python setup.py build_ext --inplace
coverage run -m pytest
coverage report -m
  
# tests/app_test.py ..                            [100%]

# ================== 2 passed in 0.02s ==================

# Name           Stmts   Miss  Cover   Missing
# --------------------------------------------
# app/_api.pyx       8      2    75%   5, 12
# app/core.py        8      0   100%
# --------------------------------------------
# TOTAL             16      2    88%