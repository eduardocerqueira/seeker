#date: 2021-12-08T17:13:20Z
#url: https://api.github.com/gists/106dc088f773087767a13e0ad02d42b8
#owner: https://api.github.com/users/mattbowen-usds

from typing import Dict
from functools import lru_cache
import censusdata
import pandas as pd
from pandas.io.formats.style import Styler
import enum


class LanguageCensusVars(str, enum.Enum):
    C16001_001E = "total speakers"
    C16001_005E = "spanish speakers"
    C16001_008E = "french, haitian, or cajun speakers"
    C16001_011E = "german or other west germanic languages speakers"
    C16001_014E = "russian, polish, or other slavic languages speakers"
    C16001_017E = "other indo-european languages speakers"
    C16001_020E = "korean speakers"
    C16001_023E = "chinese (incl. mandarin, cantonese) speakers"
    C16001_026E = "vietnamese speakers"
    C16001_029E = "tagalog (incl. filipino) speakers"
    C16001_032E = "other asian and pacific island languages speakers"
    C16001_035E = "arabic speakers"
    C16001_038E = "other and unspecified languages speakers"

    # B16009_001E = "poverty and language total"


@lru_cache
def get_frame_for_vars(CensusVars: enum.Enum) -> pd.DataFrame:
    state_data = censusdata.download(
        "acs5",
        2019,
        censusdata.censusgeo([("state", "*")]),
        [val.name for val in CensusVars],
    )

    county_data = censusdata.download(
        "acs5",
        2019,
        censusdata.censusgeo([("state", "*"), ("county", "*")]),
        [val.name for val in CensusVars],
    )

    data = pd.concat([state_data, county_data])
    data.columns = [val.value for val in CensusVars]
    data["state fips"] = data.index.map(lambda idx: idx.params()[0][-1])
    data["place name"] = data.index.map(lambda idx: idx.name)
    sort_col = [val.value for val in CensusVars][0]

    data = (
        data.set_index(["state fips", "place name"])
        .sort_values(sort_col, ascending=False)
        .sort_index(level=0, sort_remaining=False)
    )
    return data


@lru_cache
def get_state_fips_codes() -> Dict[str, str]:
    return {
        name: geo.params()[0][-1]
        for name, geo in censusdata.geographies(
            censusdata.censusgeo([("state", "*")]), "acs5", 2019
        ).items()
    }


def get_state_language_data(state_fips: str) -> Styler:
    state_data = get_frame_for_vars(LanguageCensusVars).loc[state_fips]
    return (
        (
            state_data[
                [
                    val.value
                    for val in LanguageCensusVars
                    if val.value != "total speakers"
                ]
            ].truediv(state_data["total speakers"], axis=0)
        )
        .style.applymap(lambda v: "background-color: #e6ffe6;" if v > 0.01 else None)
        .format("{:.2%}".format)
    )
