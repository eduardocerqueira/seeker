#date: 2025-12-30T16:58:20Z
#url: https://api.github.com/gists/31969ba0da87438fb311336fa3253d89
#owner: https://api.github.com/users/do-me

import duckdb

con = duckdb.connect()
con.execute("INSTALL spatial; LOAD spatial;")

input_path = '30M_sample_index.parquet'
output_path = '30M_sample_index_with_lat_lon.parquet'

# Use ST_X and ST_Y directly on the 'geometry' column
con.execute(f"""
    COPY (
        SELECT
            ST_X(geometry) AS lon,
            ST_Y(geometry) AS lat,
            * EXCLUDE (geometry)
        FROM read_parquet('{input_path}')
    ) TO '{output_path}' (FORMAT 'parquet');
""")

# To verify and load into a dataframe if needed:
df = con.execute(f"SELECT * FROM '{output_path}' LIMIT 5").df()
con.close()

df