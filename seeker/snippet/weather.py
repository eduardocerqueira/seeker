#date: 2026-01-14T17:09:52Z
#url: https://api.github.com/gists/861321548b2ed2906c24e72e434a4635
#owner: https://api.github.com/users/vrnico

import requests

url = "https://api.open-meteo.com/v1/forecast?latitude=34.0522&longitude=-118.2437&daily=temperature_2m_max,temperature_2m_min&temperature_unit=fahrenheit&timezone=America/Los_Angeles&start_date=2026-01-14&end_date=2026-01-14"

response = requests.get(url)
data = response.json()

date = data["daily"]["time"][0]
high = data["daily"]["temperature_2m_max"][0]
low = data["daily"]["temperature_2m_min"][0]

print(f"Weather for Los Angeles on {date}:")
print(f"High: {high}°F")
print(f"Low: {low}°F")