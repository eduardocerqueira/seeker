#date: 2022-03-15T16:51:28Z
#url: https://api.github.com/gists/917875679d5979405a1851d165104318
#owner: https://api.github.com/users/ccwu0918

# I create my own library to make it even easier
!pip install kora -q
from kora.selenium import wd
wd.get("https://www.website.com")
print(wd.page_source)  # results