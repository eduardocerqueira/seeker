#date: 2025-05-22T16:59:48Z
#url: https://api.github.com/gists/79e8140433424309870ea9c213fb0695
#owner: https://api.github.com/users/moredatarequired

# /// script
# dependencies = [
#   "lxml",
#   "pandas",
#   "pydantic_ai",
# ]
# ///

import sqlite3

import pandas as pd
from pydantic_ai import Agent, RunContext


def create_demo_agent() -> Agent:
    """Create a simple database query agent."""
    agent = Agent(
        "openai:gpt-4o",
        instructions=(
            "Answer the user's question using the database query tool. First inspect "
            "the available tables and scheme to ensure you understand the database "
            "well enough to be sure you can answer the question. Then execute as many   "
            "queries as you need to in order be sure you have the correct answer.",
        ),
    )

    @agent.tool
    async def query_sqlite_database(
        ctx: RunContext[sqlite3.Connection], sql_query: str
    ) -> str:
        """Execute a SQL query against the database and return the results."""
        conn = ctx.deps

        print(f"Executing query: {sql_query}")
        try:
            df = pd.read_sql_query(sql_query, conn, index_col=None)
            xml_string = df.to_xml(index=False)
        except (ValueError, sqlite3.OperationalError) as e:
            print(f"Error executing query: {e}")
            return f"Error executing query: {e}"

        print(f"Query results: {xml_string}\n")
        return xml_string

    return agent


def main():
    # Create in-memory SQLite database
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()

    # Create a simple inventory table
    cursor.execute("""
    CREATE TABLE CurrentInventory (
        id INTEGER PRIMARY KEY,
        item_name TEXT,
        quantity INTEGER
    )
    """)

    # Insert sample data
    sample_data = [(1, "Widget", 42), (2, "Gadget", 15), (3, "Doohickey", 7)]
    cursor.executemany(
        "INSERT INTO CurrentInventory (id, item_name, quantity) VALUES (?, ?, ?)",
        sample_data,
    )
    conn.commit()

    # Create and run the agent
    agent = create_demo_agent()
    question = "How many total items are there in the inventory?"
    result = agent.run_sync(question, deps=conn)

    print(f"Question: {question}")
    print(f"Answer: {result.output}")

    # Close the connection
    conn.close()


if __name__ == "__main__":
    main()

