#date: 2024-11-19T17:00:50Z
#url: https://api.github.com/gists/085cd216d9276551678bfb148e2e9dc4
#owner: https://api.github.com/users/NostraDavid

# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "polars",
# ]
# ///
from pathlib import Path
from typing import OrderedDict
import polars as pl

PROJECT = Path().cwd()

basepath = PROJECT / "jobs"

# Read the CSV file into a Polars DataFrame
df = pl.scan_csv(
    f"{basepath}/demographic_divident.csv",
    schema=OrderedDict(
        [
            ("SortOrder", pl.Int64),
            ("LocID", pl.Int64),
            ("Notes", pl.String),
            ("ISO3_code", pl.String),
            ("ISO2_code", pl.String),
            ("SDMX_code", pl.Int64),
            ("LocTypeID", pl.Int64),
            ("LocTypeName", pl.String),
            ("ParentID", pl.Int64),
            ("Location", pl.String),
            ("VarID", pl.Int64),
            ("Variant", pl.String),
            ("Time", pl.Int64),
            ("TPopulation1Jan", pl.Float64),
            ("TPopulation1July", pl.Float64),
            ("TPopulationMale1July", pl.Float64),
            ("TPopulationFemale1July", pl.Float64),
            ("PopDensity", pl.Float64),
            ("PopSexRatio", pl.Float64),
            ("MedianAgePop", pl.Float64),
            ("NatChange", pl.Float64),
            ("NatChangeRT", pl.Float64),
            ("PopChange", pl.Float64),
            ("PopGrowthRate", pl.Float64),
            ("DoublingTime", pl.Float64),
            ("Births", pl.Float64),
            ("Births1519", pl.Float64),
            ("CBR", pl.Float64),
            ("TFR", pl.Float64),
            ("NRR", pl.Float64),
            ("MAC", pl.Float64),
            ("SRB", pl.Float64),
            ("Deaths", pl.Float64),
            ("DeathsMale", pl.Float64),
            ("DeathsFemale", pl.Float64),
            ("CDR", pl.Float64),
            ("LEx", pl.Float64),
            ("LExMale", pl.Float64),
            ("LExFemale", pl.Float64),
            ("LE15", pl.Float64),
            ("LE15Male", pl.Float64),
            ("LE15Female", pl.Float64),
            ("LE65", pl.Float64),
            ("LE65Male", pl.Float64),
            ("LE65Female", pl.Float64),
            ("LE80", pl.Float64),
            ("LE80Male", pl.Float64),
            ("LE80Female", pl.Float64),
            ("InfantDeaths", pl.Float64),
            ("IMR", pl.Float64),
            ("LBsurvivingAge1", pl.Float64),
            ("Under5Deaths", pl.Float64),
            ("Q5", pl.Float64),
            ("Q0040", pl.Float64),
            ("Q0040Male", pl.Float64),
            ("Q0040Female", pl.Float64),
            ("Q0060", pl.Float64),
            ("Q0060Male", pl.Float64),
            ("Q0060Female", pl.Float64),
            ("Q1550", pl.Float64),
            ("Q1550Male", pl.Float64),
            ("Q1550Female", pl.Float64),
            ("Q1560", pl.Float64),
            ("Q1560Male", pl.Float64),
            ("Q1560Female", pl.Float64),
            ("NetMigrations", pl.Float64),
            ("CNMR", pl.Int64),
        ]
    ),
)


# Average Life Expectancy by Location
avg_life_expectancy = (
    df.group_by("Location")
    .agg(pl.col("LEx").mean().alias("AvgLifeExpectancy"))
    .sort("Location")
)

# Save average life expectancy to CSV
avg_life_expectancy.sink_csv(f"{basepath}/out/avg_life_expectancy.csv")

# Total Births and Deaths by Location
total_births_deaths = (
    df.group_by("Location")
    .agg(
        [
            pl.col("Births").sum().alias("TotalBirths"),
            pl.col("Deaths").sum().alias("TotalDeaths"),
        ]
    )
    .sort("Location")
)

# Save total births and deaths to CSV
total_births_deaths.sink_csv(f"{basepath}/out/total_births_deaths.csv")
