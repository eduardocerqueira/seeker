#date: 2022-02-22T16:54:50Z
#url: https://api.github.com/gists/16b575fd84b675eef9869b60fc0de11b
#owner: https://api.github.com/users/jimmy-law

import requests
import urllib.parse
from itertools import chain

class SportsDB:
    base_url = "https://www.thesportsdb.com/api/v1/json/2/"

    # constructor
    def __init__(self):
        pass
    
    # given dictionary of parameters return url query string
    def __make_query_string(self, params):
        query_string = "?"
        for param_name, param_value in params.items():
            query_string += param_name + "=" + param_value + "&"
        return query_string
    
    # make API call - handle any http errors
    def call(self, method, params, payload_name):
        request_url = SportsDB.base_url + method + self.__make_query_string(params)
        response = requests.get(request_url)
        return response.json().get(payload_name, [])

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
            "at_local": json["strTimestamp"]
        } for json in payload]
        return games

    def get_last_5_games_for_league(self, league):
        teams = self.get_teams_in_league(league)
        last_5_by_team = [self.get_last_5_games(team["id"]) for team in teams]
        last_5_by_team = list(chain.from_iterable(last_5_by_team))
        return last_5_by_team