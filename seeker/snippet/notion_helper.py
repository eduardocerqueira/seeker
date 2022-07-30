#date: 2022-07-30T18:53:54Z
#url: https://api.github.com/gists/c90d3f94e6c33c6c90c5d5c36aaf7127
#owner: https://api.github.com/users/ddyjis

from typing import TypedDict
from typing import Union

import requests
import yaml
from django.conf import settings


class TextContent(TypedDict):
    content: str


class Text(TypedDict):
    text: TextContent


class Name(TypedDict):
    name: str


class TitleField(TypedDict):
    title: list[Text]


class MultiSelectField(TypedDict):
    multi_select: list[Name]


class RichTextField(TypedDict):
    rich_text: list[Text]


Field = Union[MultiSelectField, RichTextField, TitleField]


class NotionDatabase:
    def __init__(self) -> None:
        self.agent = requests.Session()
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"w "**********"i "**********"t "**********"h "**********"  "**********"o "**********"p "**********"e "**********"n "**********"( "**********"s "**********"e "**********"t "**********"t "**********"i "**********"n "**********"g "**********"s "**********". "**********"D "**********"A "**********"T "**********"A "**********"_ "**********"D "**********"I "**********"R "**********"  "**********"/ "**********"  "**********"" "**********"n "**********"o "**********"t "**********"i "**********"o "**********"n "**********"_ "**********"s "**********"e "**********"c "**********"r "**********"e "**********"t "**********". "**********"y "**********"a "**********"m "**********"l "**********"" "**********") "**********"  "**********"a "**********"s "**********"  "**********"f "**********": "**********"
            data = yaml.load(f, Loader=yaml.FullLoader)
            self.agent.headers.update(
                {
                    "Authorization": "**********"
                    "Notion-Version": "2022-02-22",
                }
            )
            self.id = data.get("database", "")
        self.base_url = "https://api.notion.com/v1"

    def list(self) -> list[dict[str, str]]:
        response = self.agent.post(f"{self.base_url}/databases/{self.id}/query").json()
        data = [
            {
                key: self._get_property_value(value)
                for key, value in result.get("properties", {}).items()
            }
            for result in response.get("results", [])
        ]
        while response.get("next_cursor"):
            response = self.agent.post(response.get("next_cursor")).json()
            data.extend(
                [
                    {
                        key: self._get_property_value(value)
                        for key, value in result.get("properties", {}).items()
                    }
                    for result in response.get("results", [])
                ]
            )

        return data

    def add(self, data: dict, title: str) -> requests.Response:
        return self.agent.post(
            f"{self.base_url}/pages",
            json={"parent": {"database_id": self.id}, "properties": self._to_property(data, title)},
        )

    @staticmethod
    def _to_field(value: Union[str, list[str]], is_title: bool) -> Field:
        if is_title:
            return {"title": [{"text": {"content": str(value)}}]}
        if isinstance(value, list):
            return {"multi_select": [{"name": item} for item in value]}
        # TODO: handle other types
        return {"rich_text": [{"text": {"content": value}}]}

    @staticmethod
    def _to_property(data: dict, title: str) -> dict[str, Field]:
        return {key: NotionDatabase._to_field(value, key == title) for key, value in data.items()}

    @staticmethod
    def _get_property_value(property: dict) -> Union[str, list[str]]:
        _type = property.get("type")
        if not _type or _type not in property or not property[_type]:
            return ""
        if _type == "multi_select":
            return [item["name"] for item in property[_type]]
        # TODO: handle other types
        return property[_type][0]["plain_text"]

    def _call(self, method, url, *args, **kwargs) -> requests.Response:
        if url.startswith("/"):
            url = f"https://api.notion.com/v1{url}"
        return self.agent.request(method, url, *args, **kwargs)
