#date: 2022-12-28T16:59:29Z
#url: https://api.github.com/gists/1d3b98544eb2f4833d2fff9e04940321
#owner: https://api.github.com/users/danilo-nzyte

from typing import List
import io

import pandas as pd
from google.cloud import bigquery
from google.cloud.bigquery.schema import SchemaField


DATA = {
    "name": "Danilo",
    "age": 32,
    "date_joined": "2020-11-05",
    "location": {"country": "United Kingdom", "city": "London"},
    "years_active": [2020, 2021, 2022],
    "favourite_movies": [
        {"name": "Momento", "year": "2000"},
        {"name": "Se7en", "year": "1995"},
        {"name": "Momento", "year": "2000"},
    ],
}


def generate_bigquery_schema(df: pd.DataFrame) -> List[SchemaField]:
    TYPE_MAPPING = {
        "i": "INTEGER",
        "u": "NUMERIC",
        "b": "BOOLEAN",
        "f": "FLOAT",
        "O": "STRING",
        "S": "STRING",
        "U": "STRING",
        "M": "TIMESTAMP",
    }
    schema = []
    for column, dtype in df.dtypes.items():
        val = df[column].iloc[0]
        mode = "REPEATED" if isinstance(val, list) else "NULLABLE"

        if isinstance(val, dict) or (mode == "REPEATED" and isinstance(val[0], dict)):
            fields = generate_bigquery_schema(pd.json_normalize(val))
        else:
            fields = ()

        type = "RECORD" if fields else TYPE_MAPPING.get(dtype.kind)
        schema.append(
            SchemaField(
                name=column,
                field_type=type,
                mode=mode,
                fields=fields,
            )
        )
    return schema


def load_data_to_bq(
    client: bigquery.Client,
    data: str,
    table_id: str,
    load_config: bigquery.LoadJobConfig,
) -> int:
    load_job = client.load_table_from_file(
        io.StringIO(data), table_id, location="EU", job_config=load_config
    )
    load_job.result()  # waits for the job to complete.
    destination_table = client.get_table(table_id)
    num_rows = destination_table.num_rows
    return num_rows


if __name__ == "__main__":
    df = pd.DataFrame([DATA])
    df["date_joined"] = pd.to_datetime(df["date_joined"])
    schema = generate_bigquery_schema(df)
    json_records = df.to_json(orient="records", lines=True, date_format="iso")

    bigquery_client = bigquery.Client()
    load_config = bigquery.LoadJobConfig(
        schema=schema,
        write_disposition="WRITE_TRUNCATE",
        source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
    )
    table_id = f"dataset.table"

    num_rows = load_data_to_bq(bigquery_client, json_records, table_id, load_config)
    print(f"Successfully loaded {num_rows} to {table_id}")
