#date: 2023-09-21T17:04:50Z
#url: https://api.github.com/gists/2af37105f3fa459c0dd0a067e5109260
#owner: https://api.github.com/users/chahine-nahed

import requests
import pandas as pd


def load_availability(apikey, country, provider, start, end, granularity, fuel_types=[]):
    """
    :param apikey:
    :param country: "ISO2"
    :param provider: "bmrs" | "edf" | "eex" | "entsoe" | "nordpool" | "rte"
    :param start: "yyyy-mm-dd"
    :param end: "yyyy-mm-dd"
    :param granularity: "hourly" | "daily" | "monthly"
    :param fuel_types: "biomass" | "fossil brown coal/lignite" | "fossil coal-derived gas" | "fossil gas" | "fossil hard coal" | "fossil oil" | "fossil oil shale" | "fossil peat" | "geothermal" | "hydro pumped storage" | "hydro run-of-river and poundage" | "hydro water reservoir" | "marine" | "nuclear" | "other"
    :return: DataFrame
    """
    url = ("https://calc6.cor-e.fr/power"
           "/outages/v1/availability/series/fuel-types?"
           f"country={country}"
           f"&provider={provider}"
           f"&start={start}"
           f"&end={end}"
           f"&granularity={granularity}"
           f"{''.join('&fuel_types=' + f for f in fuel_types) if fuel_types else ''}")
    headers = {'Accept': 'application/json', 'Authorization': apikey}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    json = response.json()
    index = json['index']
    data = json['data']
    res = pd.DataFrame.from_dict(data)
    res.index = index
    return res


df = load_availability('[REDACTED]', 'DE', 'entsoe', '2023-09-21', '2023-10-22', 'daily')