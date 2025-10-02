#date: 2025-10-02T17:03:08Z
#url: https://api.github.com/gists/157c61fddb42865f2047c8042b1ca52a
#owner: https://api.github.com/users/simonw

#!/usr/bin/env python3
"""
Datasette Allow Block Matching - Pure SQLite Implementation

This demonstrates how Datasette's permission checking logic can be implemented
entirely in SQLite using JSON functions.

Based on: https://docs.datasette.io/en/stable/authentication.html

Key insight: Using a CTE to pre-extract actor values avoids issues with
deeply nested JSON function calls.
"""

import sqlite3
import json

def setup_database():
    """Create tables and sample data"""
    conn = sqlite3.connect(':memory:')
    
    conn.execute("CREATE TABLE allow_blocks (name TEXT, allow_json TEXT)")
    conn.executemany("INSERT INTO allow_blocks VALUES (?, ?)", [
        ("root_only", json.dumps({"id": "root"})),
        ("simon_or_cleopaws", json.dumps({"id": ["simon", "cleopaws"]})),
        ("developer_role", json.dumps({"roles": ["developer"]})),
        ("any_authenticated", json.dumps({"id": "*"})),
        ("allow_all", "true"),
        ("deny_all", "false"),
        ("unauthenticated_only", json.dumps({"unauthenticated": True})),
        ("multiple_keys", json.dumps({"id": ["simon", "cleopaws"], "role": "ops"})),
    ])
    
    conn.execute("CREATE TABLE test_actors (name TEXT, actor_json TEXT)")
    conn.executemany("INSERT INTO test_actors VALUES (?, ?)", [
        ("root_user", json.dumps({"id": "root"})),
        ("simon", json.dumps({"id": "simon", "name": "Simon"})),
        ("cleopaws", json.dumps({"id": "cleopaws", "roles": ["dog"]})),
        ("developer", json.dumps({"id": "dev1", "roles": ["staff", "developer"]})),
        ("ops_user", json.dumps({"id": "ops1", "role": ["ops", "staff"]})),
        ("unauthenticated", None),
        ("alice", json.dumps({"id": "alice"})),
    ])
    
    return conn

def check_permission(conn, actor_json, allow_json):
    """
    Core permission check using SQLite JSON functions.
    Returns 1 for allow, 0 for deny.
    """
    query = """
    WITH actor_values AS (
        -- Pre-extract all actor key-value pairs to avoid nested json_extract issues
        SELECT key, value, type
        FROM json_each(COALESCE(:actor_json, '{}'))
    )
    SELECT 
        CASE 
            -- Boolean allow blocks
            WHEN :allow_json = 'true' THEN 1
            WHEN :allow_json = 'false' THEN 0
            
            -- Unauthenticated actor
            WHEN :actor_json IS NULL THEN
                CASE WHEN json_extract(:allow_json, '$.unauthenticated') = 1 THEN 1 ELSE 0 END
            
            -- Object-based allow blocks (check if ANY key matches - OR logic)
            WHEN json_type(:allow_json) = 'object' THEN
                COALESCE((
                    SELECT MAX(
                        CASE 
                            -- Skip special keys
                            WHEN allow_entry.key = 'unauthenticated' THEN 0
                            
                            -- Wildcard: allow has "*" and actor has this key
                            WHEN allow_entry.value = '*' 
                                 AND EXISTS (SELECT 1 FROM actor_values WHERE key = allow_entry.key)
                            THEN 1
                            
                            -- Direct value match
                            WHEN allow_entry.type IN ('text', 'integer', 'real', 'true', 'false', 'null') THEN
                                CASE
                                    -- Actor has array: check if allow value is in actor array
                                    WHEN (SELECT type FROM actor_values WHERE key = allow_entry.key) = 'array'
                                    THEN (
                                        SELECT COUNT(*) > 0
                                        FROM json_each((SELECT value FROM actor_values WHERE key = allow_entry.key)) AS actor_val
                                        WHERE actor_val.value = allow_entry.value
                                    )
                                    -- Actor has single value: direct match
                                    ELSE allow_entry.value = (SELECT value FROM actor_values WHERE key = allow_entry.key)
                                END
                            
                            -- Array in allow block
                            WHEN allow_entry.type = 'array' THEN
                                CASE
                                    -- Actor has array: check for intersection
                                    WHEN (SELECT type FROM actor_values WHERE key = allow_entry.key) = 'array'
                                    THEN (
                                        SELECT COUNT(*) > 0
                                        FROM json_each((SELECT value FROM actor_values WHERE key = allow_entry.key)) AS actor_val
                                        WHERE EXISTS (
                                            SELECT 1
                                            FROM json_each(allow_entry.value) AS allow_val
                                            WHERE allow_val.value = actor_val.value
                                        )
                                    )
                                    -- Actor has single value: check if in allow array
                                    ELSE (
                                        SELECT COUNT(*) > 0
                                        FROM json_each(allow_entry.value) AS allow_val
                                        WHERE allow_val.value = (SELECT value FROM actor_values WHERE key = allow_entry.key)
                                    )
                                END
                            
                            ELSE 0
                        END
                    )
                    FROM json_each(:allow_json) AS allow_entry
                ), 0)
            
            ELSE 0
        END as allowed
    """
    
    result = conn.execute(query, {
        "actor_json": actor_json,
        "allow_json": allow_json
    }).fetchone()
    
    return result[0]

def main():
    conn = setup_database()
    
    print("=" * 90)
    print("Datasette Allow Block Matching - Pure SQLite Implementation")
    print("=" * 90)
    print()
    
    # Test all combinations
    actors = conn.execute("SELECT name, actor_json FROM test_actors ORDER BY rowid").fetchall()
    allow_blocks = conn.execute("SELECT name, allow_json FROM allow_blocks ORDER BY rowid").fetchall()
    
    current_actor = None
    for actor_name, actor_json in actors:
        if actor_name != current_actor:
            if current_actor:
                print()
            current_actor = actor_name
            actor_display = actor_json if actor_json else "NULL (unauthenticated)"
            print(f"\nActor: {actor_name}")
            print(f"  JSON: {actor_display}")
            print("-" * 90)
        
        for allow_name, allow_json in allow_blocks:
            allowed = check_permission(conn, actor_json, allow_json)
            status = "✓ ALLOW" if allowed else "✗ DENY "
            print(f"  {status} | {allow_name:25} | {allow_json}")
    
    # Specific test scenarios
    print("\n\n" + "=" * 90)
    print("Validation Tests")
    print("=" * 90)
    print()
    
    test_cases = [
        ("Root matches root", {"id": "root"}, {"id": "root"}, True),
        ("Simon in list", {"id": "simon"}, {"id": ["simon", "cleopaws"]}, True),
        ("Alice not in list", {"id": "alice"}, {"id": ["simon", "cleopaws"]}, False),
        ("Developer role match", {"id": "dev1", "roles": ["staff", "developer"]}, {"roles": ["developer"]}, True),
        ("Wildcard match", {"id": "anyone"}, {"id": "*"}, True),
        ("Unauth with flag", None, {"unauthenticated": True}, True),
        ("Unauth without flag", None, {"id": "root"}, False),
        ("Allow all", {"id": "anyone"}, True, True),
        ("Deny all", {"id": "root"}, False, False),
        ("OR logic - ID match", {"id": "simon"}, {"id": ["simon"], "role": "admin"}, True),
        ("OR logic - role match", {"id": "other", "role": ["admin"]}, {"id": ["simon"], "role": "admin"}, True),
    ]
    
    passed = failed = 0
    for name, actor, allow, expected in test_cases:
        actor_json = json.dumps(actor) if actor is not None else None
        allow_json = json.dumps(allow) if isinstance(allow, dict) else str(allow).lower()
        
        result = check_permission(conn, actor_json, allow_json)
        if (result == 1) == expected:
            print(f"✓ PASS | {name}")
            passed += 1
        else:
            print(f"✗ FAIL | {name} (expected {expected}, got {result})")
            failed += 1
    
    print(f"\nResults: {passed} passed, {failed} failed")
    
    print("\n" + "=" * 90)
    print("Summary")
    print("=" * 90)
    print()
    print("✓ Datasette's allow block matching works entirely in SQLite!")
    print()
    print("Key techniques:")
    print("  • CTE to pre-extract actor values (avoids nested json_extract issues)")
    print("  • json_each() for iterating JSON objects and arrays")
    print("  • Nested subqueries with EXISTS for set operations")
    print("  • COALESCE and MAX for OR logic across multiple conditions")
    print()
    
    conn.close()

if __name__ == "__main__":
    main()
