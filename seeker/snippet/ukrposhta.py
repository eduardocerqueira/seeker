#date: 2024-01-03T16:52:53Z
#url: https://api.github.com/gists/a6f812f7c5be1e6f616d63272c2b33f2
#owner: https://api.github.com/users/hyzyla

from typing import List, Optional

import requests
from pydantic import BaseModel, Field


class GetPostOfficeEntry(BaseModel):
    # Basic information
    lock_en: str = Field(alias="LOCK_EN")
    lock_ua: str = Field(alias="LOCK_UA")
    postterminal: str = Field(alias="POSTTERMINAL")
    postcode: str = Field(alias="POSTCODE")
    isautomated: str = Field(alias="ISAUTOMATED")
    is_security: str = Field(alias="IS_SECURITY")
    lock_code: str = Field(alias="LOCK_CODE")

    # Contact information
    phone: str = Field(alias="PHONE")
    postoffice_id: str = Field(alias="POSTOFFICE_ID")
    postoffice_ua: str = Field(alias="POSTOFFICE_UA")
    postoffice_ua_details: Optional[str] = Field(alias="POSTOFFICE_UA_DETAILS")

    # Location information
    longitude: str = Field(alias="LONGITUDE")
    city_katottg: str = Field(alias="CITY_KATOTTG")
    street_ua_vpz: str | None = Field(alias="STREET_UA_VPZ")
    city_ua_vpz: str = Field(alias="CITY_UA_VPZ")
    city_vpz_katottg: str = Field(alias="CITY_VPZ_KATOTTG")
    city_vpz_id: str = Field(alias="CITY_VPZ_ID")
    city_koatuu: str = Field(alias="CITY_KOATUU")
    street_id_vpz: str | None = Field(alias="STREET_ID_VPZ")
    city_vpz_koatuu: str = Field(alias="CITY_VPZ_KOATUU")
    city_ua: str = Field(alias="CITY_UA")
    lattitude: str = Field(alias="LATTITUDE")
    city_ua_type: str = Field(alias="CITY_UA_TYPE")

    # Office type information
    type_acronym: str = Field(alias="TYPE_ACRONYM")
    type_long: str = Field(alias="TYPE_LONG")
    type_id: str = Field(alias="TYPE_ID")

    # Postal information
    postindex: str = Field(alias="POSTINDEX")
    city_id: str = Field(alias="CITY_ID")
    housenumber: str | None = Field(alias="HOUSENUMBER")


class GetPostOfficiesEntry(BaseModel):
    entry: List[GetPostOfficeEntry] = Field(alias="Entry")


class GetPostOfficesRoot(BaseModel):
    entries: dict = Field(alias="Entries")


class UkrposhtaClient:
    def __init__(self, token: "**********":
        self.base_url = "https://www.ukrposhta.ua/address-classifier-ws"
        self.token = "**********"

    def get_post_offices(
        self,
        *,
        katottg: str | None = None,
        koatuu: str | None = None,
    ) -> list[GetPostOfficeEntry]:
        params = {}
        if katottg:
            params["city_katottg"] = katottg.removeprefix("UA")
        if koatuu:
            params["city_koatuu"] = koatuu
        if not params:
            raise ValueError("Either katottg or koatuu must be specified")

        response = requests.get(
            url=f"{self.base_url}/get_postoffices_by_postcode_cityid_cityvpzid",
            headers={
                "Authorization": "**********"
                "Accept": "application/json",
            },
            params=params,
        )
        response.raise_for_status()
        response_json = response.json()

        root = GetPostOfficesRoot(**response_json)
        if not root.entries:
            return []
        return GetPostOfficiesEntry(**root.entries).entryntry