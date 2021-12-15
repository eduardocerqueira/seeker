#date: 2021-12-15T17:13:06Z
#url: https://api.github.com/gists/93ad46b21839179caded05fd219f9e47
#owner: https://api.github.com/users/tomasonjo

wikifan_url = "https://matrix.fandom.com/wiki/Category:Characters_in_The_Matrix"

member_list = []
wd.get(wikifan_url)
members = wd.find_elements_by_class_name("category-page__member-link")
for m in members:
  member_list.append({'url':m.get_attribute('href'), 'name': m.text})

# Manually append Trinity
member_list.append({'url': 'https://matrix.fandom.com/wiki/Trinity', 'name': 'Trinity'})