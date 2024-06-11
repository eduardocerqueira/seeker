#date: 2024-06-11T17:07:23Z
#url: https://api.github.com/gists/c5f46a612bf498d3100936b84d693ea3
#owner: https://api.github.com/users/elijahbenizzy

import my_module
dr = (driver
  .Builder()
  .with_modules(my_module)
  .build()
)

dr.execute(["c", "a", "b"], inputs={"external_input" : 10}) # dataframe with c, a, and b
# inputs are upstream dependencies
