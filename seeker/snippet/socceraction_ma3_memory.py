#date: 2022-03-18T17:06:09Z
#url: https://api.github.com/gists/28d4668c4861b7c551a6caba3c341ba2
#owner: https://api.github.com/users/denisov-vlad

import datetime
from typing import Any
from typing import Dict
from typing import Tuple
from typing import Type

import pandas as pd
import socceraction.spadl.opta as opta
from pandera.typing import DataFrame
from socceraction.data.opta.loader import _eventtypesdf
from socceraction.data.opta.loader import OptaLoader
from socceraction.data.opta.parsers.base import _get_end_x
from socceraction.data.opta.parsers.base import _get_end_y
from socceraction.data.opta.parsers.base import assertget
from socceraction.data.opta.parsers.ma3_json import MA3JSONParser
from socceraction.data.opta.schema import OptaCompetitionSchema
from socceraction.data.opta.schema import OptaEventSchema
from socceraction.data.opta.schema import OptaGameSchema
from socceraction.data.opta.schema import OptaPlayerSchema
from socceraction.data.opta.schema import OptaTeamSchema


class MA3MemoryParser:
    def __init__(self):
        pass

    @staticmethod
    def extract_events(events_list: list) -> Dict[Tuple[str, int], Dict[str, Any]]:
        events = {}
        for element in events_list:
            timestamp = assertget(element, "timeStamp")
            # timestamp = self._convert_timestamp(timestamp_string)

            qualifiers = {
                int(q["qualifierId"]): q.get("value")
                for q in element.get("qualifier", [])
            }
            start_x = float(assertget(element, "x"))
            start_y = float(assertget(element, "y"))
            end_x = _get_end_x(qualifiers) or start_x
            end_y = _get_end_y(qualifiers) or start_y

            event_id = int(assertget(element, "id"))
            game_id = assertget(element, "matchId")

            event = dict(
                # Fields required by the base schema
                game_id=game_id,
                event_id=event_id,
                period_id=int(assertget(element, "periodId")),
                team_id=assertget(element, "contestantId"),
                player_id=element.get("playerId"),
                type_id=int(assertget(element, "typeId")),
                # Fields required by the opta schema
                timestamp=timestamp,
                minute=int(assertget(element, "timeMin")),
                second=int(assertget(element, "timeSec")),
                outcome=bool(int(element.get("outcome", 1))),
                start_x=start_x,
                start_y=start_y,
                end_x=end_x,
                end_y=end_y,
                qualifiers=qualifiers,
                # Optional fields
                assist=bool(int(element.get("assist", 0))),
                keypass=bool(int(element.get("keyPass", 0))),
            )
            events[(game_id, event_id)] = event
        return events


class OptaMemoryLoader:
    def __init__(self, data: list, parser: Type[MA3MemoryParser]):
        self.data = data
        self.parser = parser()
        self.events_df = self.events()

    def competitions(self):
        raise NotImplementedError()

    def games(self):
        return self.events_df.game_id.unique().tolist()

    def teams(self):
        raise NotImplementedError()

    def players(self):
        raise NotImplementedError()

    def events(self):

        data = self.parser.extract_events(self.data)

        events = (
            pd.DataFrame(list(data.values()))
            .merge(_eventtypesdf, on="type_id", how="left")
            .sort_values(["game_id", "period_id", "minute", "second", "timestamp"])
            .reset_index(drop=True)
        )

        # sometimes pre-match events has -3, -2 and -1 seconds
        events.loc[events.second < 0, "second"] = 0
        events = events.sort_values(
            ["game_id", "period_id", "minute", "second", "timestamp"]
        )

        # deleted events has wrong datetime which occurs OutOfBoundsDatetime error
        events = events[events.type_id != 43]
        events = events[
            ~(
                (events.timestamp < datetime.datetime(1900, 1, 1))
                | (events.timestamp > datetime.datetime(2100, 1, 1))
            )
        ]

        return events.pipe(DataFrame[OptaEventSchema])
