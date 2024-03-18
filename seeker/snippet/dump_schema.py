#date: 2024-03-18T17:07:29Z
#url: https://api.github.com/gists/a6213306ec1f7421017d1180a780a7ec
#owner: https://api.github.com/users/jgarland79

from google.cloud import bigquery

def list_datasets_tables_schemas_data():
    client = bigquery.Client()
    project_id = client.project

    datasets = list(client.list_datasets())
    print(f"Datasets in project {project_id}:")
    for dataset in datasets:
        print(f"\tDataset: {dataset.dataset_id}")

        tables = list(client.list_tables(dataset.reference))
        print(f"\tTables in dataset {dataset.dataset_id}:")
        for table in tables:
            print(f"\t\tTable: {table.table_id}")

            table_ref = dataset.table(table.table_id)
            table = client.get_table(table_ref)
            print(f"\t\tSchema of table {table.table_id}:")
            for field in table.schema:
                print(f"\t\t\t{field.name}: {field.field_type}")

            # Print the first few rows of data from the table
            print(f"\t\tData sample from {project_id}.{dataset.dataset_id}.{table.table_id}:")
            query = f"SELECT * FROM `{project_id}.{dataset.dataset_id}.{table.table_id}` LIMIT 5"
            query_job = client.query(query)
            results = query_job.result()

            for row in results:
                print(f"\t\t\t{dict(row)}")

if __name__ == "__main__":
    list_datasets_tables_schemas_data()