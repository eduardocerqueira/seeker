#date: 2022-02-22T17:03:39Z
#url: https://api.github.com/gists/57d964017ad2cba8512cee4bd660356c
#owner: https://api.github.com/users/jimmy-law

import requests
import urllib.parse
from itertools import chain
import datetime as dt
import pytz

class SportsDB:
    base_url = "https://www.thesportsdb.com/api/v1/json/2/"

    # constructor
    def __init__(self, timezone="Europe/London"):
        self.display_timezone = pytz.timezone(timezone)
    
    # given dictionary of parameters return url query string
    def __make_query_string(self, params):
        query_string = "?"
        for param_name, param_value in params.items():
            query_string += param_name + "=" + param_value + "&"
        return query_string
    
    # convert a SportsDB timestamp to localized display string
    def to_display_string(self, timestamp_str):
        # handle some timestamp strings not have the offset due to data quality issue
        if len(timestamp_str)==25:
            dt_orig = dt.datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S%z") 
        else:
            dt_orig = dt.datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S")
        ts_orig = dt_orig.timestamp()
        dt_display = dt.datetime.fromtimestamp(ts_orig, self.display_timezone)
        return dt_display.strftime("%d %b %Y %H:%M")


    # make API call - handle any http errors
    def call(self, method, params, payload_name):
        request_url = SportsDB.base_url + method + self.__make_query_string(params)
        response = requests.get(request_url)
        if response.status_code == 200:
            payload = response.json().get(payload_name, [])
        elif response.status_code == 429:
            # for this demo we simply catch it here - we could easily implement re-try after some specified time
            payload = "throttled by SportsDB"
        else:
            payload = "other error"
        return payload

    def get_teams_in_league(self, league_name):
        method = "search_all_teams.php"
        payload_name = "teams"
        params = {"l": urllib.parse.quote(league_name.encode("utf8"))}
        payload = self.call(method, params, payload_name)
        teams = [{"id": json["idTeam"], "name": json["strTeam"]} for json in payload]
        return teams

    def get_last_5_games(self, team_id):
        method = "eventslast.php"
        payload_name = "results"
        params = {"id": str(team_id)}
        payload = self.call(method, params, payload_name)
        games = [{
            "home": json["strHomeTeam"], 
            "home_score": json["intHomeScore"], 
            "away": json["strAwayTeam"], 
            "away_score": json["intAwayScore"],
            "at_local": json["strTimestamp"],
            "at_display": self.to_display_string(json["strTimestamp"])
        } for json in payload]
        return games

    def get_last_5_games_for_league(self, league):
        teams = self.get_teams_in_league(league)
        last_5_by_team = [self.get_last_5_games(team["id"]) for team in teams]
        last_5_by_team = list(chain.from_iterable(last_5_by_team))
        return last_5_by_team