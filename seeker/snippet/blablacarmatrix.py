#date: 2024-12-16T17:01:06Z
#url: https://api.github.com/gists/3387f971f5717054e7cee6e018c4ce7e
#owner: https://api.github.com/users/Romern

import requests
import datetime
import click
import tqdm
from rich.console import Console
from rich.table import Table

search_url = "https://edge.blablacar.de/trip/search/v7"
location_get_url = 'https://edge.blablacar.de/location/suggestions'
headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:133.0) Gecko/20100101 Firefox/133.0",
    "Accept": "application/json",
    "Accept-Language": "de-DE",
    "Content-Type": "application/json",
    "Authorization": "Bearer 693493b9-ca1b-4947-94a3-546093caba7b",
    "x-client": "SPA|1.0.0",
    "x-currency": "EUR",
    "x-locale": "de_DE",
    "x-visitor-id": "87810daa-0ff5-4b02-b0f1-21614b266e69"
}

def search(from_location: str, to_location: str, departure_date: datetime.date, requested_seats = 1):
    url_params = {
        "from_place_id": get_location(from_location),
        "to_place_id": get_location(to_location),
        "departure_date": departure_date.isoformat(),
        "search_uuid": "95d73ca6-09af-4a69-a56c-d84c3cddf16b",
        "requested_seats":"1",
        "search_origin":"SEARCH"
    }

    return requests.get(search_url, params=url_params, headers=headers).json()

def get_location(location_name: str) -> str:
    url_params = {
        "with_user_history": "true",
        "locale": "de_DE",
        "query": location_name
    }
    return requests.get(location_get_url, params=url_params, headers=headers).json()[0]['id']

def print_search_range(from_location: str, to_location: str, from_date: datetime.date, to_date: datetime.date, requested_seats = 1):
    table = Table(title=f"BlaBlaCar Trips from {from_location} to {to_location} on {from_date.isoformat()} to {to_date.isoformat()}")

    table.add_column("Departure Time")
    table.add_column("From Location")
    table.add_column("To Location")
    table.add_column("Price")

    

    for day in tqdm.trange((to_date - from_date).days + 1, desc="Retrieving trips for days", leave=False):
        search_results = search(from_location, to_location, from_date + datetime.timedelta(days=day), requested_seats)
        for trip in search_results['trips']:
            table.add_row(trip['waypoints'][0]['departure_datetime'], 
                          f"{trip['waypoints'][0]['place']['city']}, {trip['waypoints'][0]['place']['address']}", 
                          f"{trip['waypoints'][-1]['place']['city']}, {trip['waypoints'][-1]['place']['address']}",
                          trip['price_details']['price'])

    console = Console()
    console.print(table)

@click.command()
@click.argument('from_location')
@click.argument('to_location')
@click.argument('from_date')
@click.argument('to_date')
@click.option("--requestedseats", default=1)
def main(from_location: str, to_location: str, from_date: str, to_date: str, requestedseats):
    print_search_range(from_location, to_location, datetime.date.fromisoformat(from_date), datetime.date.fromisoformat(to_date), requested_seats=requestedseats)

if __name__ == "__main__":
    main()