#date: 2023-08-29T17:01:21Z
#url: https://api.github.com/gists/5c5c000c46acf06507b6b3577cbeb70b
#owner: https://api.github.com/users/dlozeve

import polars as pl
import altair as alt

# https://data.sncf.com/explore/dataset/meilleurs-temps-des-parcours-des-trains/information/
DATA_FILE = "meilleurs-temps-des-parcours-des-trains.csv"

# https://en.wikipedia.org/wiki/File:France_TGV.png
LGVs = pl.DataFrame(
    {
        "line": [
            "LGV Nord",
            "LGV Est",
            "LGV Sud-Est",
            "LGV Rhônes-Alpes",
            "LGV Méditerranée",
            "LGV Atlantique",
            "LGV Sud Europe Atlantique",
            "LGV Bretagne-Pays de la Loire",
        ],
        "year": [
            1993,
            2007,
            1983,
            1994,
            2001,
            1990,
            2017,
            2017,
        ],
    }
).select("line", pl.col("year").cast(str).str.strptime(pl.Date, format="%Y"))


def load_data() -> pl.DataFrame:
    return (
        pl.read_csv(DATA_FILE, separator=";")
        .select(
            pl.col("Relations")
            .str.split_exact(" - ", 1)
            .struct.rename_fields(["start", "end"]),
            pl.col("Année").cast(str).str.strptime(pl.Date, format="%Y").alias("year"),
            pl.col("Temps estimé en minutes").alias("duration"),
        )
        .unnest("Relations")
    )


def plot_durations(durations: pl.DataFrame, start: str, ends: list[str]):
    start = start.upper()
    ends = [end.upper() for end in ends]
    df = (
        durations.filter((pl.col("start") == start) & (pl.col("end").is_in(ends)))
        .select(pl.col("end").str.to_titlecase(), "year", pl.col("duration") / 60)
        .sort("end", "year")
    )

    base = alt.Chart(df.to_pandas()).encode(
        alt.Color("end", title="Destination").legend(None)
    )
    lines = base.mark_line(clip=True).encode(
        alt.X("year", title="Year").scale(domain=("1948", "2019")),
        alt.Y("duration", title="Duration (hours)")
        .scale(domain=(0, 11))
        .axis(values=list(range(12))),
    )
    last_duration = (
        base.mark_circle()
        .encode(
            alt.X("last_year['year']:T"),
            alt.Y("last_year['duration']:Q"),
        )
        .transform_aggregate(last_year="argmax(year)", groupby=["end"])
    )
    names = last_duration.mark_text(align="left", dx=4, fontSize=14).encode(text="end")
    # lgvs = (
    #     alt.Chart(LGVs.to_pandas())
    #     .encode(alt.X("year"))
    #     .mark_rule(strokeDash=(8, 4), opacity=0.5)
    # )
    return (
        (lines + last_duration + names)
        .properties(
            title=alt.Title(
                "Travel time by train from Paris to major French cities", fontSize=20
            ),
            width=800,
            height=600,
        )
        .configure_axis(
            titleFontSize=20,
            labelFontSize=18,
        )
        .configure_legend(
            titleFontSize=20,
            labelFontSize=18,
        )
    )


def main():
    durations = load_data()
    durations.write_csv("durations.csv")
    ends = [
        "Lille",
        "Strasbourg",
        "Lyon",
        "Marseille",
        "Rennes",
        "Bordeaux",
    ]
    chart = plot_durations(durations, start="Paris", ends=ends)
    chart.save("all.html")


if __name__ == "__main__":
    main()
