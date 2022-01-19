#date: 2022-01-19T17:04:49Z
#url: https://api.github.com/gists/10d913363d79b6cdef30a9e7bca3dfa4
#owner: https://api.github.com/users/phreakin

import requests
import pandas as pd
from bs4 import BeautifulSoup as BS
import csv


def yearly_stats(season=2021):
    BASE_URL = f"https://www.pro-football-reference.com/years/{season}/fantasy.htm"

    res = requests.get(BASE_URL)

    soup = BS(res.content, "html.parser")

    table = soup.find("table", {"id": "fantasy"})

    df = pd.read_html(str(table))[0]

    df.columns = df.columns.droplevel(level=0)

    df["PassingTD"] = df["TD"].iloc[:, 0]
    df["PassingYds"] = df["Yds"].iloc[:, 0]
    df["RushingTD"] = df["Yds"].iloc[:, 1]
    df["RushingYds"] = df["Yds"].iloc[:, 1]
    df["ReceivingTD"] = df["TD"].iloc[:, 2]
    df["ReceivingYds"] = df["Yds"].iloc[:, 2]

    df["PassingAtt"] = df["Att"].iloc[:, 0]
    df["RushingAtt"] = df["Att"].iloc[:, 1]

    df = df.rename({"FantPos": "Pos"}, axis=1)

    df = df[
        [
            "Player",
            "Tm",
            "Pos",
            "Age",
            "PassingAtt",
            "Cmp",
            "PassingYds",
            "PassingTD",
            "RushingAtt",
            "RushingYds",
            "RushingTD",
            "Tgt",
            "Rec",
            "ReceivingYds",
            "ReceivingTD",
            "FL",
            "2PM",
        ]
    ]

    df = df.fillna(0)

    df = df.loc[df["Player"] != "Player"]

    for column in df.columns[4:]:
        df[column] = df[column].astype(int)

    return df


def get_top_target_share(df, pos="WR", min_receptions=0, n=100):
    df = df.merge(
        df.groupby("Tm")["Tgt"].sum(), how="left", on="Tm", suffixes=("_ind", "_team")
    )

    df["Tgt_share"] = df["Tgt_ind"] / df["Tgt_team"]

    df = df.loc[(df["Pos"] == pos) & (df["Rec"] >= min_receptions)]

    df = df.sort_values(by="Tgt_share", ascending=False)

    df["Tgt_share_rank"] = df["Tgt_share"].rank(ascending=False, method="min")

    with pd.option_context("display.max_rows", None):
        df = df[["Player", "Tm", "Tgt_ind", "Rec", "Tgt_share", "Tgt_share_rank"]]
        print(df.head(n))


df = yearly_stats(2021)

get_top_target_share(df, pos="RB", min_receptions=10, n=100)

get_top_target_share(df, pos="WR", min_receptions=10, n=100)

get_top_target_share(df, pos="TE", min_receptions=10, n=100)
