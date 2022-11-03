#date: 2022-11-03T17:00:42Z
#url: https://api.github.com/gists/4676e93f505c9fb9071f6be7d1fc14c9
#owner: https://api.github.com/users/cpcloud

from pathlib import Path

import ibis
import pandas as pd
from ibis import _


def pandas(d):
    movies = pd.read_parquet(f"{d}/movies.parquet")
    ratings = pd.read_parquet(f"{d}/ratings.parquet").assign(
        timestamp=lambda df: pd.to_datetime(df.timestamp, unit="s")
    )
    links = pd.read_parquet(f"{d}/links.parquet")
    rated_movies = pd.merge(
        pd.merge(movies, ratings, on="movieId"), links, on="movieId"
    )
    names = pd.read_csv(f"{d}/name.basics.tsv", sep="\t", na_values="\\N").assign(
        primaryProfession=lambda df: df.primaryProfession.str.contains("actor|actress")
    )
    titles = pd.read_csv(f"{d}/title.basics.tsv", sep="\t", na_values="\\N")

    # principal cast/crew
    # only actors/actresses
    actors = pd.read_csv(f"{d}/title.principals.tsv.gz", sep="\t", na_values="\\N")
    actors = actors.loc[actors.category.isin(["actor", "actress"])]

    # assign actor names to titles
    named_actors = pd.merge(pd.merge(actors, names, on="nconst"), titles, on="tconst")

    # bring in the ratings
    rated_movies = rated_movies.assign(
        imdbId_tconst=lambda df: "tt" + df.imdbId.astype("string")
    )
    rated_actors = pd.merge(
        named_actors, rated_movies, left_on="tconst", right_on="imdbId_tconst"
    )
    return rated_actors.groupby("primaryName").rating.mean()


def duckdb(d):
    movies = ibis.read_parquet(f"{d}/movies.parquet")
    ratings = ibis.read_parquet(f"{d}/ratings.parquet").mutate(
        timestamp=_.timestamp.cast("timestamp")
    )
    links = ibis.read_parquet(f"{d}/links.parquet")
    rated_movies = movies.join(ratings, "movieId").join(links, "movieId")
    names = ibis.read_csv(
        f"{d}/name.basics.tsv",
        table_name="names",
        quote="",
        nullstr="\\N",
    ).filter(
        _.primaryProfession.contains("actor") | _.primaryProfession.contains("actress")
    )
    titles = ibis.read_csv(
        f"{d}/title.basics.tsv", table_name="titles", quote="", nullstr="\\N"
    )

    # principal cast/crew
    # only actors/actresses
    actors = ibis.read_csv(
        f"{d}/title.principals.tsv.gz",
        table_name="principals",
        quote="",
        nullstr="\\N",
    ).filter(_.category.isin(["actor", "actress"]))

    # assign actor names to titles
    named_actors = actors.join(names, "nconst").join(titles, "tconst")

    # bring in the ratings
    rated_actors = named_actors.join(
        rated_movies,
        named_actors.tconst == "tt" + rated_movies.imdbId.cast("string"),
    )
    return (
        rated_actors.group_by(name=_.primaryName)
        .aggregate(avg_rating=_.rating.mean(), n=_.count())
        .order_by(_.n.desc())
    )


if __name__ == "__main__":
    ibis.options.interactive = True
    d = Path(__file__).parent
    joined = duckdb(d)
    print(joined)
