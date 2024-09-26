#date: 2024-09-26T17:08:35Z
#url: https://api.github.com/gists/6eb009a8f3f7460beaa283238e219aab
#owner: https://api.github.com/users/nathanjones4323

### This example fetches daily weather forecasts for a list of CBSAs (Core-Based Statistical Areas) using the Open-Meteo API and processes the data into a pandas DataFrame for further analysis.
import pandas as pd
import requests

# Example CBSA data - you can replace this with your full dataset
cbsa_data = [
    {
        "name": "Athens-Clarke County, GA",
        "latitude": 33.9439840,
        "longitude": -83.2138965,
    },
    {
        "name": "Atlanta-Sandy Springs-Alpharetta, GA",
        "latitude": 33.6937280,
        "longitude": -84.3999113,
    },
    {
        "name": "Atlantic City-Hammonton, NJ",
        "latitude": 39.4693555,
        "longitude": -74.6337591,
    },
    # Add more CBSAs as needed...
]

# Open-Meteo API base URL for daily forecast
open_meteo_url = "https://api.open-meteo.com/v1/forecast"


# Function to fetch daily weather for a given latitude and longitude
def fetch_weather(latitude, longitude):
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "timezone": "America/New_York",  # Adjust to a suitable timezone if necessary
    }
    response = requests.get(open_meteo_url, params=params)
    return response.json()


# List to store weather results
weather_results = []

# Loop through the CBSA data and fetch weather for each
for cbsa in cbsa_data:
    weather_data = fetch_weather(cbsa["latitude"], cbsa["longitude"])
    weather_results.append(
        {
            "cbsa_name": cbsa["name"],
            "latitude": cbsa["latitude"],
            "longitude": cbsa["longitude"],
            "forecast": weather_data.get("daily", {}),
        }
    )

# Convert to pandas DataFrame for easier processing and analysis
weather_df = pd.DataFrame(weather_results)

# Expand each list into separate rows to match time and other forecast values
expanded_weather_df = weather_df.explode(["forecast"])

# Extract the 'daily' forecast lists and convert them into individual columns
expanded_weather_df = weather_df.explode(["forecast"])
daily_forecast = weather_df["forecast"].apply(pd.Series)

# Explode each forecast parameter (time, temperature, etc.) into rows
weather_df = weather_df.join(daily_forecast).explode(
    ["time", "temperature_2m_max", "temperature_2m_min", "precipitation_sum"]
)

# Combine the original DataFrame with the expanded forecast data
weather_df = pd.concat([weather_df, daily_forecast], axis=1)

# Expand the list columns into separate rows
weather_df = weather_df.explode("temperature_2m_min").reset_index(drop=True)

print(weather_df)