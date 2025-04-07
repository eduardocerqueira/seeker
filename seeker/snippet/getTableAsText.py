#date: 2025-04-07T16:56:57Z
#url: https://api.github.com/gists/8384fb67e5776716ddf1eaa90b6c18d6
#owner: https://api.github.com/users/Ruchi12377

import psycopg2
import os

# DB connection information (Supabase local environment)
conn = psycopg2.connect(
    dbname="postgres",
    user="postgres",
    password= "**********"
    host="127.0.0.1",
    port="54322"
)

cur = conn.cursor()

# Get table and column information
cur.execute("""
    SELECT 
        t.table_schema || '.' || t.table_name AS table_name,
        c.column_name,
        c.data_type,
        CASE 
            WHEN c.column_name IN (
                SELECT ccu.column_name 
                FROM information_schema.table_constraints tc
                JOIN information_schema.constraint_column_usage ccu ON tc.constraint_name = ccu.constraint_name
                WHERE tc.constraint_type = 'PRIMARY KEY' 
                AND tc.table_schema = t.table_schema 
                AND tc.table_name = t.table_name
            ) THEN 'PK'
            ELSE ''
        END AS key_type
    FROM 
        information_schema.tables t
    JOIN 
        information_schema.columns c ON t.table_schema = c.table_schema AND t.table_name = c.table_name
    WHERE 
        t.table_schema = 'public'
        AND t.table_type = 'BASE TABLE'
    ORDER BY 
        table_name, c.ordinal_position
""")

# Store table and column information
table_data = []

for row in cur.fetchall():
    table_name, column_name, data_type, key_type = row
    table_data.append([table_name, column_name, data_type, key_type])

# Get views information
cur.execute("""
    SELECT 
        v.table_schema || '.' || v.table_name AS view_name,
        c.column_name,
        c.data_type,
        v.view_definition
    FROM 
        information_schema.views v
    JOIN 
        information_schema.columns c ON v.table_schema = c.table_schema AND v.table_name = c.table_name
    WHERE 
        v.table_schema = 'public'
    ORDER BY 
        view_name, c.ordinal_position
""")

view_definitions = {}

for row in cur.fetchall():
    view_name, column_name, data_type, view_definition = row
    
    # Store view definition only once per view
    if view_name not in view_definitions:
        view_definitions[view_name] = view_definition

# Extract schema as SQL DDL statements
print("\nExtracting database schema as SQL...")
schema_sql = []

# Get table creation SQL
cur.execute("""
    SELECT 
        'CREATE TABLE ' || table_schema || '.' || table_name || ' (' ||
        string_agg(
            column_name || ' ' || data_type || 
            CASE 
                WHEN character_maximum_length IS NOT NULL THEN '(' || character_maximum_length || ')'
                ELSE ''
            END ||
            CASE 
                WHEN is_nullable = 'NO' THEN ' NOT NULL'
                ELSE ''
            END,
            ', '
        ) || ');' as create_statement
    FROM 
        information_schema.columns
    WHERE 
        table_schema = 'public'
        AND table_name IN (SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND table_type = 'BASE TABLE')
    GROUP BY 
        table_schema, table_name
""")

for row in cur.fetchall():
    schema_sql.append(row[0])

# Get primary key constraints
cur.execute("""
    SELECT
        'ALTER TABLE ' || tc.table_schema || '.' || tc.table_name || 
        ' ADD CONSTRAINT ' || tc.constraint_name || ' PRIMARY KEY (' ||
        string_agg(kcu.column_name, ', ') || ');' as pk_statement
    FROM
        information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu ON tc.constraint_name = kcu.constraint_name
    WHERE
        tc.constraint_type = 'PRIMARY KEY'
        AND tc.table_schema = 'public'
    GROUP BY
        tc.table_schema, tc.table_name, tc.constraint_name
""")

for row in cur.fetchall():
    schema_sql.append(row[0])

# Get foreign key constraints
cur.execute("""
    SELECT
        'ALTER TABLE ' || tc.table_schema || '.' || tc.table_name ||
        ' ADD CONSTRAINT ' || tc.constraint_name || ' FOREIGN KEY (' ||
        kcu.column_name || ') REFERENCES ' || ccu.table_schema || '.' ||
        ccu.table_name || '(' || ccu.column_name || ');' as fk_statement,
        tc.table_schema || '.' || tc.table_name AS source_table,
        kcu.column_name AS source_column,
        ccu.table_schema || '.' || ccu.table_name AS target_table,
        ccu.column_name AS target_column
    FROM
        information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu ON tc.constraint_name = kcu.constraint_name
        JOIN information_schema.constraint_column_usage ccu ON tc.constraint_name = ccu.constraint_name
    WHERE
        tc.constraint_type = 'FOREIGN KEY'
        AND tc.table_schema = 'public'
        AND ccu.table_schema = 'public'
""")

fk_data = cur.fetchall()
for row in fk_data:
    schema_sql.append(row[0])

# Get view creation SQL
for view_name, definition in view_definitions.items():
    schema_sql.append(f"-- View: {view_name}\n{definition};")

# Add comments to show foreign key relationships more clearly
schema_sql.append("\n-- Foreign Key Relationships Summary")
schema_sql.append("-- Format: SourceTable.SourceColumn -> TargetTable.TargetColumn")
for row in fk_data:
    _, source_table, source_col, target_table, target_col = row
    source_table_simple = source_table.split('.')[-1]
    target_table_simple = target_table.split('.')[-1]
    schema_sql.append(f"-- {source_table_simple}.{source_col} -> {target_table_simple}.{target_col}")

# Write schema to file
schema_path = os.path.abspath("database_schema.sql")
with open(schema_path, 'w', encoding='utf-8') as f:
    f.write("\n\n".join(schema_sql))

print(f"Database schema saved as SQL to: database_schema.sql")

# Close the database connection
cur.close()
conn.close()

print("\nDatabase structure extraction complete.")