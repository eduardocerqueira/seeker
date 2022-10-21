#date: 2022-10-21T17:19:01Z
#url: https://api.github.com/gists/4f24b55ad45dbc20368f6dd15970d119
#owner: https://api.github.com/users/merve-ozturk

country = {"Germany" : "DE", "Japan" : "JP", "Netherlands" : "NL", "Qatar" : "QA"}

if "Switzerland" not in country.keys():
  country["Switzerland"] = "CH"

print(country)
#{'Germany': 'DE', 'Japan': 'JP', 'Netherlands': 'NL', 'Qatar': 'QA', 'Switzerland': 'CH'}