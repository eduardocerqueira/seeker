#date: 2023-09-25T17:00:21Z
#url: https://api.github.com/gists/73a292e18e0acb9568bcde25ebfd4204
#owner: https://api.github.com/users/dataqualityuk

%python
# Import necessary modules 
from pyspark.sql import SparkSession

# Create Spark session
spark = SparkSession.builder.getOrCreate() 

# Get list of all databases
databases = [x for x in spark.catalog.listDatabases() if x.name.__contains__('sl1')]

# Print number of databases found
print(f"Number of databases to process: {len(databases)}")  

table_names = ["table1","table2","table3"]

# Loop through list of table names
for table_name in table_names:

  # Print current table  
  print(f"Processing table: {table_name}")
  
  # Initialize variables to store database names and columns
  db_names = []
  columns = None
  
  # Loop through databases
  for db in databases:

    # Try to read table 
    try:
      df = spark.table(f"{db.name}.{table_name}")

      # Get common column names  
      if columns is None:
        columns = set(df.columns) 
      else:
        columns = columns.intersection(set(df.columns))

      # Append database name  
      db_names.append(db.name)
    
    # Print error if table not found
    except:
      print(f"Cannot read spark table: {table_name} database: {db.name}")
      
  # Print number of databases found
  print(f"Number of databases for {table_name}: {len(db_names)}")  

  # Convert columns to string
  columns = ", ".join(list(columns))

  # Create SQL view if databases were found
  if db_names:

    # Create first part of view 
    sql_create = f"CREATE OR REPLACE VIEW Combined.{table_name} as SELECT {columns} FROM {db_names[0]}.{table_name}"
    
    # Remove first database already used
    del db_names[0] 

    # Add additional unions
    for db in db_names:
      sql_create += f"\n UNION ALL SELECT {columns} FROM {db}.{table_name}"

    # Execute SQL 
    spark.sql(sql_create)

  # Print message if no databases found
  else:
    print("No databases to process")