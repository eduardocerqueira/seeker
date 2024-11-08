#date: 2024-11-08T16:51:42Z
#url: https://api.github.com/gists/fefde8b7a61aeef4f338a26374e4e42e
#owner: https://api.github.com/users/mrjsj

class FabricDuckDBSession():

    def __init__(self, config: dict = {}):
        self._registered_tables = []
        self._connection = duckdb.connect(config=config)

    @property
    def connection(self):
        return self._connection

    def _attach_lakehouse(self, lakehouse: str) -> None:
        self._connection.sql(f"""
            ATTACH ':memory:' AS {lakehouse};
        """)        

    def _create_fabric_lakehouse_secret(self, lakehouse: "**********":
        
        access_token = "**********"
        
        self._connection.sql(f"""
            USE {lakehouse};                     
            CREATE OR REPLACE SECRET fabric_lakehouse_secret (
                TYPE AZURE,
                PROVIDER ACCESS_TOKEN,
                ACCESS_TOKEN '{access_token}'
            )
        """)

    def _register_lakehouse_tables(self, workspace_id: str, lakehouse_id: str, lakehouse_name: str) -> None:

        tables = notebookutils.lakehouse.listTables(lakehouse_name, data_workspace_id)

        for table in tables:
            self._connection.sql(f"""
                CREATE OR REPLACE VIEW main.{table["name"]} AS
                SELECT 
                    * 
                FROM
                    delta_scan('{table["location"]}')
            """)

            is_table_registered = any(
                registered_table for registered_table in self._registered_tables 
                if registered_table["workspace_id"] == workspace_id
                and registered_table["lakehouse_id"] == lakehouse_id
                and registered_table["table_name"] == table["name"] 
            )

            if not is_table_registered:

                table_information = {
                    "workspace_id": workspace_id,
                    "lakehouse_id": lakehouse_id,
                    "lakehouse_name": lakehouse_name,
                    "table_name": table["name"],
                    "table_location": table["location"]
                }

                self._registered_tables.append(table_information)

 
    def register_workspace_lakehouses(self, workspace_id: str, lakehouses: str | list[str] = None):
        
        if isinstance(lakehouses, str):
            lakehouses = [lakehouses]

        for lakehouse in lakehouses:

            lakehouse_properties = notebookutils.lakehouse.getWithProperties(
                lakehouse,
                workspace_id
            )

            is_schema_enabled = lakehouse_properties.get("properties").get("defaultSchema") is not None
            lakehouse_id = lakehouse_properties.get("id")
            
            if is_schema_enabled:
                raise Exception(f"""
                    The lakehouse `{lakehouse}` is using the schema-enabled preview feature.\n
                    This utility class does support schema-enabled lakehouses (yet).
                """)
            
            self._attach_lakehouse(lakehouse)
            self._create_fabric_lakehouse_secret(lakehouse)
            self._register_lakehouse_tables(workspace_id, lakehouse_id, lakehouse)

    def print_lakehouse_catalog(self):
        query = """
            SELECT 
                table_catalog as lakehouse_name,
                table_schema as schema_name,
                table_name
            FROM information_schema.tables
            ORDER BY table_catalog, table_schema, table_name
        """
        
        results = self._connection.sql(query).fetchall()
        
        current_lakehouse = None
        current_lakehouse_schema = None
        
        for lakehouse_name, schema_name, table_name in results:

            if current_lakehouse != lakehouse_name:
                current_lakehouse = lakehouse_name
                print(f"📁 Database: {lakehouse_name}")
                
            lakehouse_schema = (lakehouse_name, schema_name)
            if current_lakehouse_schema != lakehouse_schema:
                current_lakehouse_schema = lakehouse_schema
                print(f"  └─📂 Schema: {schema_name}")
                
            print(f"     ├─📄 {table_name}")



    def write(self, df, full_table_name: str, workspace_id: str = None, *args, **kwargs):

        table_parts = full_table_name.split(".")

        if len(table_parts) != 2:
            raise Exception("The parameter `table_name` must consist of two parts, i.e. `<lakehouse_name>.<table_name>`.")
        
        lakehouse_name = table_parts[0]
        table_name = table_parts[1]

        if not workspace_id:
            workspace_ids = list(set([registed_table["workspace_id"] for registed_table in self._registered_tables]))

            if len(workspace_ids) > 1:
                raise Exception("The FabricDuckDBSession has registered multiple workspaces, so `workspace_id` must be supplied.")
            
            workspace_id = workspace_ids[0]

        table_information = [
            table for table in self._registered_tables
            if table["workspace_id"] == workspace_id
            and table["lakehouse_name"] == lakehouse_name
        ][0]

        lakehouse_id = table_information["lakehouse_id"]


        table_uri = f"abfss://{data_workspace_id}@onelake.dfs.fabric.microsoft.com/{lakehouse_id}/Tables/{table_name}"

        write_deltalake(
            table_or_uri=table_uri,
            data=df,
            *args,
            **kwargs,
        )

        table_information = [
            table for table in self._registered_tables
            if table["workspace_id"] == workspace_id
            and table["lakehouse_name"] == lakehouse_name
            and table["table_name"] == table_name
        ]

        if len(table_information) == 0:
            self._connection.sql(f"""
                CREATE OR REPLACE VIEW {lakehouse_name}.main.{table_name} AS
                SELECT 
                    * 
                FROM
                    delta_scan('{table_uri}')
            """)

'{table_uri}')
            """)

