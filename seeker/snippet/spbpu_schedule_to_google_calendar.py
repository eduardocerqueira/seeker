#date: 2024-08-29T17:04:19Z
#url: https://api.github.com/gists/335367f343d28ca3f6612f965b33bbd3
#owner: https://api.github.com/users/iwsylit

"""
Simple script for copying SPbPU lessons to Google Calendar.

Before usage:
- find the id of your group
- get Google Calendar API credentials
- fill env variables (GROUP_ID, GOOGLE_CALENDAR_ID, GOOGLE_SECRETS_FILE, GOOGLE_CREDENTIALS_FILE)
- pip install requests==2.32.3 google-api-python-client==2.142.0 google-auth-oauthlib==2.0.0
"""

import logging
import os
import pickle
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from operator import itemgetter
from typing import Any, Self
from zoneinfo import ZoneInfo

import requests
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")


def getenv(key: str) -> str:
    var = os.getenv(key)

    if var is None:
        raise ValueError(f"There is no env variable {key}")

    return var


class DateTime(datetime):
    def __new__(cls, *args: Any, timezone: ZoneInfo = ZoneInfo("Etc/GMT-3"), **kwargs: Any) -> Self:
        instance = super().__new__(cls, *args, **kwargs)

        return instance.replace(tzinfo=timezone)

    @classmethod
    def from_datetime(cls, datetime: datetime) -> Self:
        return cls(
            datetime.year,
            datetime.month,
            datetime.day,
            datetime.hour,
            datetime.minute,
            datetime.second,
            datetime.microsecond,
        )

    @classmethod
    def from_iso(cls, date: str) -> Self:
        return cls.from_datetime(datetime.fromisoformat(date))

    @classmethod
    def from_date_time(cls, date: str, time: str) -> Self:
        return cls.from_datetime(datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M"))

    @classmethod
    def today(cls) -> Self:
        return cls.now().replace(hour=0, minute=0, second=0, microsecond=0)

    def monday(self) -> Self:
        days_to_monday = timedelta(days=self.weekday())
        monday = self - days_to_monday

        return monday.replace(hour=0, minute=0, second=0, microsecond=0)

    def isodateformat(self) -> str:
        return self.strftime("%Y-%m-%d")


class Event(ABC):
    def __init__(self, id: str, name: str, location: str, description: str, start: DateTime, end: DateTime) -> None:
        self.id = id
        self.name = name
        self.location = location
        self.description = description
        self.start = start
        self.end = end

    @abstractmethod
    def construct(cls, event: dict) -> Self:
        pass

    def googleformat(self) -> dict:
        return {
            "summary": self.name,
            "location": self.location,
            "description": self.description,
            "start": {
                "dateTime": self.start.isoformat(),
            },
            "end": {
                "dateTime": self.end.isoformat(),
            },
        }

    def __eq__(self, value: object) -> bool:
        return self.__hash__() == value.__hash__()

    def __hash__(self) -> int:
        return hash((self.name, self.location, self.description, self.start, self.end))

    def __repr__(self) -> str:
        return f"{self.name}; {self.start.time()}-{self.end.time()}; {self.description}"


class GoogleEvent(Event):
    @classmethod
    def construct(cls, event: dict) -> Self:
        return cls(
            id=event["id"],
            name=event["summary"],
            location=event["location"],
            description=event["description"],
            start=DateTime.from_iso(event["start"]["dateTime"]),
            end=DateTime.from_iso(event["end"]["dateTime"]),
        )


class PolyEvent(Event):
    @classmethod
    def construct(cls, event: dict) -> Self:
        auditory = event["auditories"][0]

        teacher = ", ".join(map(itemgetter("full_name"), event["teachers"]))
        lms = "LMS: " + event["lms_url"] if event["lms_url"] else ""
        webinar = "Webinar: " + event["webinar_url"] if event["webinar_url"] else ""

        return cls(
            id="",
            name=event["subject"],
            location=f"{auditory['building']['name']}, ауд. {auditory['name']}",
            description="\n".join([teacher, lms, webinar]).strip(),
            start=DateTime.from_date_time(event["date"], event["time_start"]),
            end=DateTime.from_date_time(event["date"], event["time_end"]),
        )


class Calendar(ABC):
    def __init__(self) -> None:
        super().__init__()
        logging.info(f"Connecting to {self.__class__.__name__}")

    @abstractmethod
    def list_week_events(self, start: DateTime) -> set[Event]:
        pass


class GoogleCalendar(Calendar):
    _scopes = ["https://www.googleapis.com/auth/calendar"]
    _secrets_file = "**********"
    _credentials_file = getenv("GOOGLE_CREDENTIALS_FILE")
    _calendar_id = getenv("GOOGLE_CALENDAR_ID")

    def __init__(self) -> None:
        super().__init__()

        if not os.path.exists(self._credentials_file):
            flow = "**********"
            creds = flow.run_local_server(port=0)

            with open(self._credentials_file, "wb") as f:
                pickle.dump(creds, f)
        else:
            with open(self._credentials_file, "rb") as f:
                creds = pickle.load(f)

        self.api = build("calendar", "v3", credentials=creds)

    def list_week_events(self, start: DateTime) -> set[Event]:
        end = start + timedelta(days=6)

        events = (
            self.api.events()
            .list(
                calendarId=self._calendar_id,
                timeMin=start.isoformat(),
                timeMax=end.isoformat(),
            )
            .execute()
        )["items"]

        return set(map(GoogleEvent.construct, events))

    def create(self, event: Event) -> None:
        logging.info(f"Create event {event}")
        self.api.events().insert(calendarId=self._calendar_id, body=event.googleformat()).execute()

    def remove(self, event: Event) -> None:
        logging.info(f"Remove event {event}")
        self.api.events().delete(calendarId=self._calendar_id, eventId=event.id).execute()


class PolyCalendar(Calendar):
    _group_id = getenv("GROUP_ID")

    def list_week_events(self, start: DateTime) -> set[Event]:
        response = requests.get(self._url(start))
        response.raise_for_status()
        schedule = response.json()

        events = []

        for day in schedule["days"]:
            for event in day["lessons"]:
                event["date"] = day["date"]

                events.append(PolyEvent.construct(event))

        return set(events)

    def _url(self, start: DateTime) -> str:
        return f"https://ruz.spbstu.ru/api/v1/ruz/scheduler/{self._group_id}?date={start.isodateformat()}"


if __name__ == "__main__":
    logging.info("Begin working")

    poly_calendar = PolyCalendar()
    google_calendar = GoogleCalendar()

    for week in range(4):
        start = DateTime.today().monday() + timedelta(days=7 * week)

        logging.info(f"Parse {start.isodateformat()} week")

        poly_events = poly_calendar.list_week_events(start)
        google_events = google_calendar.list_week_events(start)

        new_events = poly_events.difference(google_events)
        expired_events = google_events.difference(poly_events)

        logging.debug(f"Poly events: {list(poly_events)}")
        logging.debug(f"Google events: {list(google_events)}")
        logging.debug(f"New events: {list(new_events)}")
        logging.debug(f"Expired events: {list(expired_events)}")

        if not new_events and not expired_events:
            logging.info("There is no updates")
        elif not new_events:
            logging.info("There is no new events")
        elif not expired_events:
            logging.info("There is no expired events")

        for event in expired_events:
            google_calendar.remove(event)

        for event in new_events:
            google_calendar.create(event)
nt)

        for event in new_events:
            google_calendar.create(event)
