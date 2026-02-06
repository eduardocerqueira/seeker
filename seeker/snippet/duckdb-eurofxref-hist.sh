#date: 2026-02-06T17:34:11Z
#url: https://api.github.com/gists/b328b8d45585593ae50b74796b985869
#owner: https://api.github.com/users/atrakic

duckdb :memory:  -c "CREATE TABLE eurofxref_hist AS SELECT * FROM read_csv_auto(\"https://csvbase.com/calpaterson/eurofxref-hist\"); select * from eurofxref_hist;"