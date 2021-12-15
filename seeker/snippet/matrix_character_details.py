#date: 2021-12-15T17:13:45Z
#url: https://api.github.com/gists/d04a98c19dac50bd9f9d2f1ec004cb30
#owner: https://api.github.com/users/tomasonjo

for m in member_list:
  wd.get(m['url'])
  elements = wd.find_elements_by_class_name("pi-data")
  for e in elements:
    try:
      label = e.find_element_by_tag_name("h3")
      value = e.find_element_by_tag_name("div")
      m[label.text] = value.text
    except:
      pass