#date: 2021-12-15T17:15:21Z
#url: https://api.github.com/gists/fb4ba864a9c6b0238160dd01ae2c10b1
#owner: https://api.github.com/users/tomasonjo

entity_query = """
UNWIND $data as row
CREATE (c:Character)
SET c += row
"""
with driver.session() as session:
  session.run(entity_query, {'data': member_list})