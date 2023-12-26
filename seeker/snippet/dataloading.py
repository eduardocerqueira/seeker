#date: 2023-12-26T17:00:47Z
#url: https://api.github.com/gists/845ac6c8f9a1dcbc31ab14f20d44eeb1
#owner: https://api.github.com/users/taner45

import duckdb

atp_duck = duckdb.connect('atp.duck.db', read_only=True)

atp_duck.sql("INSTALL httpfs")
atp_duck.sql("LOAD httpfs")

csv_files = [
    f"https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{year}.csv"
    for year in range(1968,2024)
]
atp_duck.execute("""
CREATE OR REPLACE TABLE matches AS 
SELECT * FROM read_csv_auto($1, types={'winner_seed': 'VARCHAR', 'loser_seed': 'VARCHAR'})
""", [csv_files])