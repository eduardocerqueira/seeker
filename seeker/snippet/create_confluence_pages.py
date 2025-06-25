#date: 2025-06-25T17:12:48Z
#url: https://api.github.com/gists/7b1e74a331ef4db61dad6e57ca8f1a86
#owner: https://api.github.com/users/philerooski

import os
import random
import logging
import argparse
import toml
import snowflake.connector
from atlassian import Confluence

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

# Helper functions for Confluence
def get_or_create_page(space: str, client: Confluence, title: str, parent_id: str, **kwargs) -> str:
    """
    Retrieve a page by title under parent_id in the given space using CQL search;
    create it if it doesn't exist. Returns the page_id.
    """
    cql = f'title = "{title}" AND ancestor = {parent_id} AND space = "{space}"'
    res = client.cql(cql, limit=1)
    if res.get('results'):
        page_id = res['results'][0]['id']
        log.info(f"Found existing Confluence page '{title}' (ID: {page_id}) via CQL search")
        return page_id
    new = client.create_page(
        space=space,
        title=title,
        body="",
        parent_id=parent_id,
        representation='wiki',
        type=kwargs.get("type", "page"),
    )
    log.info(f"Created blank Confluence page '{title}' (ID: {new['id']})")
    return new['id']

# Initialize Snowflake connector session and cursor
def init_snowflake_cursor(connection_name: str, default_role: str = 'SYNAPSE_DATA_WAREHOUSE_ANALYST'):
    """
    Use ~/.snowflake/connections.toml with the specified profile name to establish a connection.
    Applies a default role if none is specified in the profile.
    Returns (connection, cursor).
    """
    config_file = os.path.expanduser('~/.snowflake/connections.toml')
    conn = snowflake.connector.connect(
        config_path=config_file,
        connection_name=connection_name,
        role=default_role
    )
    return conn, conn.cursor()

# Argument parsing for Confluence and Snowflake settings
def parse_args():
    parser = argparse.ArgumentParser(description='Sync Snowflake metadata to Confluence')
    parser.add_argument('--confluence-url', default=os.getenv('CONFLUENCE_URL', 'https://sagebionetworks.jira.com/wiki'),
                        help='Base URL for Confluence')
    parser.add_argument('--confluence-user', default=os.getenv('CONFLUENCE_USER'),
                        help='Confluence username')
    parser.add_argument('--confluence-api-token', default= "**********"
                        help= "**********"
    parser.add_argument('--confluence-space', default=os.getenv('CONFLUENCE_SPACE', 'DPE'),
                        help='Confluence space key')
    parser.add_argument('--root-page-id', default='4194992145',
                        help='Root page ID under which pages will be created')
    parser.add_argument('--sf-connection-name', default='default',
                        help='Profile name under [connections] in ~/.snowflake/connections.toml')
    return parser.parse_args()

# Function to fetch schemas and their tables
def fetch_schema_tables(cursor) -> dict:
    schema_tables = {}
    cursor.execute("SHOW SCHEMAS IN DATABASE synapse_data_warehouse")
    for row in cursor.fetchall():
        schema = row[1]
        if schema.upper() in ('INFORMATION_SCHEMA', 'PUBLIC'):
            continue
        cursor.execute(f"SHOW TABLES IN SCHEMA synapse_data_warehouse.{schema}")
        tables = [r[1] for r in cursor.fetchall()]
        schema_tables[schema] = tables
    return schema_tables

# Function to fetch table metadata
def fetch_table_metadata(cursor, schema: str, table: str) -> (str, list):
    # Description
    cursor.execute(
        "SELECT comment FROM synapse_data_warehouse.information_schema.tables "
        "WHERE table_schema = %s AND table_name = %s",
        (schema, table)
    )
    row = cursor.fetchone()
    description = row[0] if row and row[0] else 'No description.'
    # Columns
    cursor.execute(
        "SELECT column_name, comment FROM synapse_data_warehouse.information_schema.columns "
        "WHERE table_schema = %s AND table_name = %s "
        "ORDER BY ordinal_position",
        (schema, table)
    )
    columns = cursor.fetchall()
    return description, columns

def get_random_animal_emoji():
    """
    Returns a random animal emoji as a Unicode character.
    """
    animal_emojis = [
        "🐶", "🐱", "🐭", "🐹", "🐰",
        "🦊", "🐻", "🐼", "🐨", "🐯",
        "🦁", "🐮", "🐷", "🐸", "🐵",
        "🐔", "🐧", "🐦", "🐤", "🦆",
        "🦉", "🦇", "🐺", "🦄", "🐝",
        "🐛", "🦋", "🐞", "🦀", "🐙",
        "🦐", "🦑", "🐟", "🐠", "🐬",
        "🐳", "🐋"
    ]
    return random.choice(animal_emojis)

# Example usage:
if __name__ == "__main__":
    print(get_random_animal_emoji())


# Function to build Confluence page body in wiki format
def build_page_body(description: str, columns: list) -> str:
    """
    Returns wiki markup body with description and table of column comments.
    """
    # Header for description and columns
    parts = []
    parts.append('h2. Description')
    parts.append(description)
    parts.append('')
    parts.append('h2. Columns')
    # Table header
    parts.append('||column||comment||')
    # Table rows
    for col, comment in columns:
        parts.append(f'|{col.lower()}|{comment or ""}|')
    # Join with newlines
    return '\n'.join(parts)

# Main execution
def main():
    args = parse_args()
    confluence = Confluence(
        url=args.confluence_url,
        username=args.confluence_user,
        password= "**********"
    )
    space = args.confluence_space
    root_id = args.root_page_id
    ctx, cs = init_snowflake_cursor(args.sf_connection_name)

    try:
        schema_tables = fetch_schema_tables(cs)
        for schema, tables in schema_tables.items():
            emoji_prefix = get_random_animal_emoji()
            schema_title = f"{emoji_prefix} {schema}"
            schema_id = get_or_create_page(space, confluence, schema_title, root_id)
            tables_id = get_or_create_page(space, confluence, f'{emoji_prefix} Tables', schema_id, type='folder')
            for table in tables:
                table_title = f"{emoji_prefix} {table}"
                print(f"getting id of {table_title}")
                page_id = get_or_create_page(space, confluence, table_title, tables_id)
                print(f"getting metadata of {table_title}")
                description, cols = fetch_table_metadata(cs, schema, table)
                print(f"building body of {table_title}")
                body = build_page_body(description, cols)
                print(f"updating {table_title}")
                confluence.update_page(
                    page_id=page_id,
                    title=table_title,
                    body=body,
                    parent_id=tables_id,
                    representation='wiki'
                )
    finally:
        cs.close()
        ctx.close()
    log.info("Finished syncing Snowflake metadata to Confluence.")

if __name__ == "__main__":
    main()

")

if __name__ == "__main__":
    main()

