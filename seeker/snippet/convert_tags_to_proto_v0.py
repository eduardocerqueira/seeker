#date: 2026-03-06T17:19:53Z
#url: https://api.github.com/gists/2942de405a1b45f0452e2ec4000de989
#owner: https://api.github.com/users/mohsen-dd

#!/usr/bin/env python3
"""
xp-tag-conversion v0 — Convert dietary metadata tags from Snowflake to labels-store protobuf.

Queries Snowflake for per-item dietary metadata (IS_HEALTHY, IS_VEGETARIAN) from the
Webster pipeline, filters out invalid items (alcohol, beverages, supplies, etc.),
aggregates to store-level GMV percentages, and produces a gzipped protobuf Batch file
matching the labels-store ingest/v1 schema for S3 ingestion.

Protobuf schema:
    github.com/deliveroo/transport-models-go/.../consumer/labels_store/ingest/v1/batch.proto

Output format:
    Batch {
      records: [EntityLabels {
        entity_drn_id: "partner-<restaurant_id>"
        labels: { "<namespace>": NamespacedLabels { slugs: ["healthy", "vegetarian"] } }
      }]
      created_at: <timestamp>
    }

S3 ingestion guide:
    github.com/deliveroo/labels-store/blob/main/S3_INGESTION_GUIDE.md
    Key format: ingest/v1/partner/<uuid>/<batch_uuid>.gz

Prerequisites:
    pip install snowflake-connector-python protobuf

Authentication:
    The script reads Snowflake credentials from ~/.snowflake/connections.toml
    (default connection). You can override with --sf-token, --sf-account, etc.
    Supports both PROGRAMMATIC_ACCESS_TOKEN and externalbrowser (SSO) auth.

Usage:
    # Test run — 20 stores, 30% threshold
    python convert_tags_to_proto_v0.py --threshold 30 --limit 20

    # Full run — 50% threshold, output to specific directory
    python convert_tags_to_proto_v0.py --threshold 50 --output-dir ./output

    # Override Snowflake connection
    python convert_tags_to_proto_v0.py --threshold 50 \\
        --sf-account DOORDASH --sf-user JANE.DOE --sf-warehouse ADHOC

Options:
    --threshold   Min GMV % to qualify for a label (default: 50)
    --namespace   Protobuf namespace for the labels (default: dietary-webster)
    --limit       Limit number of stores fetched (for testing)
    --output-dir  Directory for output files (default: current dir)
    --sf-account  Snowflake account identifier (default: from connections.toml)
    --sf-user     Snowflake username (default: from connections.toml)
    --sf-warehouse Snowflake warehouse (default: from connections.toml)
    --sf-database  Snowflake database (default: from connections.toml)
    --sf-schema    Snowflake schema (default: from connections.toml)
    --sf-token     Snowflake PAT (default: "**********"

Note:
    The namespace "dietary-webster" must be registered in common.proto before
    production ingestion will succeed. Current valid namespaces: metadata,
    family_time, cuisine, food, collection, locale_broad, locale_niche,
    food_broad, food_niche.
"""

import argparse
import base64
import gzip
import os
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

import snowflake.connector
from google.protobuf import descriptor_pool, message_factory
from google.protobuf.timestamp_pb2 import Timestamp

# ---------------------------------------------------------------------------
# Protobuf: embedded serialized FileDescriptorProto for batch.proto
# Generated via: protoc --descriptor_set_out=... batch.proto
# This avoids needing a separate _pb2.py file.
# ---------------------------------------------------------------------------

_BATCH_PROTO_SERIALIZED = (
    b'\n\x0b\x62\x61tch.proto\x12;deliveroo.protobuf.messages.consumer'
    b'.labels_store.ingest.v1\x1a\x1fgoogle/protobuf/timestamp.proto'
    b'\"\x93\x01\n\x05\x42\x61tch\x12Z\n\x07records\x18\x01 \x03(\x0b\x32'
    b'I.deliveroo.protobuf.messages.consumer.labels_store.ingest.v1.'
    b'EntityLabels\x12.\n\ncreated_at\x18\x02 \x01(\x0b\x32\x1a.google'
    b'.protobuf.Timestamp\"\x97\x02\n\x0c\x45ntityLabels\x12\x15\n\r'
    b'entity_drn_id\x18\x01 \x01(\t\x12\x65\n\x06labels\x18\x03 \x03'
    b'(\x0b\x32U.deliveroo.protobuf.messages.consumer.labels_store'
    b'.ingest.v1.EntityLabels.LabelsEntry\x1a|\n\x0bLabelsEntry\x12\x0b'
    b'\n\x03key\x18\x01 \x01(\t\x12\\\n\x05value\x18\x02 \x01(\x0b\x32M'
    b'.deliveroo.protobuf.messages.consumer.labels_store.ingest.v1.'
    b'NamespacedLabels:\x02\x38\x01J\x04\x08\x02\x10\x03R\x05slugs\"!'
    b'\n\x10NamespacedLabels\x12\r\n\x05slugs\x18\x01 \x03(\tBhB\n'
    b'BatchOuterZZgithub.com/deliveroo/transport-models-go/types/'
    b'messages/consumer/labels_store/ingest/v1;v1b\x06proto3'
)

_NS = "deliveroo.protobuf.messages.consumer.labels_store.ingest.v1"

_pool = descriptor_pool.Default()
_pool.AddSerializedFile(_BATCH_PROTO_SERIALIZED)

Batch = message_factory.GetMessageClass(_pool.FindMessageTypeByName(f"{_NS}.Batch"))
EntityLabels = message_factory.GetMessageClass(_pool.FindMessageTypeByName(f"{_NS}.EntityLabels"))
NamespacedLabels = message_factory.GetMessageClass(_pool.FindMessageTypeByName(f"{_NS}.NamespacedLabels"))


# ---------------------------------------------------------------------------
# Snowflake SQL
# ---------------------------------------------------------------------------

SETUP_SQL = """\
--------------------------------------------------------------------------------
-- 1) Extract primary/top dish tag (NaN-safe), up to level 4
--------------------------------------------------------------------------------
CREATE OR REPLACE TEMPORARY TABLE dish_top_tags AS
SELECT
  TRY_CAST(TO_VARCHAR(t.PLATFORM_ITEM_ID) AS NUMBER) AS ITEM_ID,
  COALESCE(
    t.pred:top_candidates[0]:hierarchical_tags::STRING,
    t.pred:hierarchical_tags::STRING
  ) AS TOP_HIERARCHICAL_TAG,
  NULLIF(SPLIT_PART(
    COALESCE(t.pred:top_candidates[0]:hierarchical_tags::STRING,
             t.pred:hierarchical_tags::STRING), ' > ', 1), '') AS LEVEL_1_TAG,
  NULLIF(SPLIT_PART(
    COALESCE(t.pred:top_candidates[0]:hierarchical_tags::STRING,
             t.pred:hierarchical_tags::STRING), ' > ', 2), '') AS LEVEL_2_TAG,
  NULLIF(SPLIT_PART(
    COALESCE(t.pred:top_candidates[0]:hierarchical_tags::STRING,
             t.pred:hierarchical_tags::STRING), ' > ', 3), '') AS LEVEL_3_TAG,
  NULLIF(SPLIT_PART(
    COALESCE(t.pred:top_candidates[0]:hierarchical_tags::STRING,
             t.pred:hierarchical_tags::STRING), ' > ', 4), '') AS LEVEL_4_TAG
FROM (
  SELECT PLATFORM_ITEM_ID,
         TRY_PARSE_JSON(REPLACE(PREDICTION, 'NaN', 'null')) AS pred
  FROM STATIC.RX_METADATA_OUTPUT_WEBSTER_DISH_INTL_ROO_FULL
  WHERE PLATFORM_ITEM_ID IS NOT NULL
) t;

--------------------------------------------------------------------------------
-- 2) Build invalid/exclusion item lists
--------------------------------------------------------------------------------
CREATE OR REPLACE TEMPORARY TABLE invalid_items_1 AS
SELECT DISTINCT TRY_CAST(TO_VARCHAR(PLATFORM_ITEM_ID) AS NUMBER) AS ITEM_ID
FROM STATIC.RX_METADATA_OUTPUT_WEBSTER_INTL_ROO_FULL t
WHERE PLATFORM_ITEM_ID IS NOT NULL
  AND (t.CUISINE IS NULL
       OR NULLIF(TRIM(t.CUISINE), '[]') IS NULL
       OR TRIM(t.CUISINE) = '')
  AND (t.PORTION_SIZE IS NULL
       OR LOWER(TRIM(COALESCE(t.PORTION_SIZE, ''))) IN ('', 'none', 'na'))
  AND (t.DISH_PREPARATION_METHOD IS NULL
       OR LOWER(TRIM(COALESCE(t.DISH_PREPARATION_METHOD, '')))
          IN ('', 'none', 'na'))
  AND COALESCE(TRY_CAST(TO_VARCHAR(t.ESTIMATED_CALORIES) AS NUMBER), 0) = 0;

CREATE OR REPLACE TEMPORARY TABLE invalid_items_2 AS
SELECT DISTINCT ITEM_ID
FROM dish_top_tags
WHERE LEVEL_1_TAG = 'Non-Alcoholic Beverages'
  AND COALESCE(LEVEL_2_TAG, '') <> 'Smoothies';

CREATE OR REPLACE TEMPORARY TABLE invalid_items_3 AS
SELECT DISTINCT ITEM_ID
FROM dish_top_tags
WHERE COALESCE(LEVEL_1_TAG, '') IN (
  'Alcohol', 'Restaurant Supplies',
  'Side Dishes & Condiments', 'Sauces, Condiments & Dressings');

CREATE OR REPLACE TEMPORARY TABLE all_invalid_items AS
SELECT ITEM_ID FROM invalid_items_1
UNION
SELECT ITEM_ID FROM invalid_items_2
UNION
SELECT ITEM_ID FROM invalid_items_3;

--------------------------------------------------------------------------------
-- 3) Build filtered item metadata (IS_HEALTHY, IS_VEGETARIAN)
--------------------------------------------------------------------------------
CREATE OR REPLACE TEMPORARY TABLE item_metadata_raw AS
SELECT
  TRY_CAST(TO_VARCHAR(PLATFORM_ITEM_ID) AS NUMBER) AS ITEM_ID,
  MAX(CASE WHEN TRY_CAST(IS_HEALTHY AS BOOLEAN) THEN 1 ELSE 0 END) AS IS_HEALTHY,
  MAX(CASE WHEN TRY_CAST(IS_VEGETARIAN AS BOOLEAN) THEN 1 ELSE 0 END) AS IS_VEGETARIAN
FROM STATIC.RX_METADATA_OUTPUT_WEBSTER_INTL_ROO_FULL
WHERE PLATFORM_ITEM_ID IS NOT NULL
GROUP BY TRY_CAST(TO_VARCHAR(PLATFORM_ITEM_ID) AS NUMBER);

CREATE OR REPLACE TEMPORARY TABLE item_metadata_filtered AS
SELECT r.*
FROM item_metadata_raw r
LEFT JOIN all_invalid_items a ON r.ITEM_ID = a.ITEM_ID
WHERE a.ITEM_ID IS NULL;
"""

STORE_STATS_SQL = """\
SELECT
    s.RESTAURANT_ID,
    s.RESTAURANT_NAME,
    s.COUNTRY_NAME,
    SUM(s.GMV_LOCAL_CURRENCY) AS TOTAL_GMV,
    ROUND(
        100.0 * SUM(CASE WHEN m.IS_HEALTHY = 1
                         THEN s.GMV_LOCAL_CURRENCY ELSE 0 END)
        / NULLIF(SUM(s.GMV_LOCAL_CURRENCY), 0), 2) AS HEALTHY_PCT,
    ROUND(
        100.0 * SUM(CASE WHEN m.IS_VEGETARIAN = 1
                         THEN s.GMV_LOCAL_CURRENCY ELSE 0 END)
        / NULLIF(SUM(s.GMV_LOCAL_CURRENCY), 0), 2) AS VEGETARIAN_PCT
FROM MOHSENMOLLANOORI.RX_ROO_ITEMS_SALES_90D_AGG s
LEFT JOIN item_metadata_filtered m ON s.ITEM_ID = m.ITEM_ID
GROUP BY s.RESTAURANT_ID, s.RESTAURANT_NAME, s.COUNTRY_NAME
"""


# ---------------------------------------------------------------------------
# Snowflake connection
# ---------------------------------------------------------------------------

def _read_connections_toml() -> Dict:
    """Read ~/.snowflake/connections.toml and return the default connection config."""
    toml_path = os.path.expanduser("~/.snowflake/connections.toml")
    if not os.path.exists(toml_path):
        return {}
    try:
        import tomllib
        with open(toml_path, "rb") as f:
            cfg = tomllib.load(f)
    except ImportError:
        # Python < 3.11 fallback
        try:
            import tomli
            with open(toml_path, "rb") as f:
                cfg = tomli.load(f)
        except ImportError:
            import pip._vendor.tomli as _tomli
            with open(toml_path, "r") as f:
                cfg = _tomli.loads(f.read())
    conn_name = cfg.get("default_connection_name", "default")
    return cfg.get(conn_name, {})


def connect_snowflake(account: str, user: str, warehouse: str, database: str,
                      schema: "**********": Optional[str] = None):
    """Connect to Snowflake. Uses PAT if token is provided, otherwise externalbrowser SSO."""
    print(f"Connecting to Snowflake ({account}) as {user} ...")
    kwargs = dict(
        account=account, user=user, warehouse=warehouse,
        database=database, schema=schema,
    )
 "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********": "**********"
        kwargs["authenticator"] = "**********"
        kwargs["token"] = "**********"
    else:
        kwargs["authenticator"] = "externalbrowser"
    conn = snowflake.connector.connect(**kwargs)
    print("Connected.")
    return conn


# ---------------------------------------------------------------------------
# SQL execution
# ---------------------------------------------------------------------------

def run_setup(conn):
    """Execute the temp-table setup statements one by one."""
    cur = conn.cursor()
    count = 0
    for stmt in SETUP_SQL.split(";"):
        # Skip segments that are only comments/whitespace
        lines = stmt.strip().splitlines()
        has_sql = any(
            ln.strip() and not ln.strip().startswith("--") for ln in lines
        )
        if not has_sql:
            continue
        count += 1
        print(f"  Running setup statement {count} ...")
        cur.execute(stmt.strip())
    cur.close()


def fetch_store_stats(conn, limit: Optional[int] = None) -> List[Dict]:
    """Return store-level stats as list of dicts."""
    sql = STORE_STATS_SQL
    if limit:
        sql += f"\nLIMIT {limit}"
    cur = conn.cursor()
    cur.execute(sql)
    cols = [desc[0] for desc in cur.description]
    rows = [dict(zip(cols, row)) for row in cur.fetchall()]
    cur.close()
    return rows


# ---------------------------------------------------------------------------
# Protobuf construction
# ---------------------------------------------------------------------------

def build_batch(rows: List[Dict], threshold: float, namespace: str):
    """Build a protobuf Batch from store-level rows.

    For each store, assigns "healthy" and/or "vegetarian" slugs under the
    given namespace if the store's GMV percentage meets the threshold.
    entity_drn_id is set to "partner-<restaurant_id>".
    """
    batch = Batch()
    now = datetime.now(timezone.utc)
    ts = Timestamp()
    ts.FromDatetime(now)
    batch.created_at.CopyFrom(ts)

    skipped = 0
    for row in rows:
        slugs = []
        if (row.get("HEALTHY_PCT") or 0) >= threshold:
            slugs.append("healthy")
        if (row.get("VEGETARIAN_PCT") or 0) >= threshold:
            slugs.append("vegetarian")
        if not slugs:
            skipped += 1
            continue

        entity = batch.records.add()
        entity.entity_drn_id = f"partner-{row['RESTAURANT_ID']}"
        entity.labels[namespace].slugs.extend(slugs)

    print(f"  Built {len(batch.records)} records "
          f"({skipped} stores below threshold, skipped).")
    return batch


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def serialize_and_write(batch, output_dir: str):
    """Serialize batch to raw .pb and gzipped .pb.gz files."""
    raw = batch.SerializeToString()
    compressed = gzip.compress(raw)
    batch_uuid = str(uuid.uuid4())

    bin_path = os.path.join(output_dir, f"{batch_uuid}.pb")
    gz_path = os.path.join(output_dir, f"{batch_uuid}.pb.gz")

    with open(bin_path, "wb") as f:
        f.write(raw)
    with open(gz_path, "wb") as f:
        f.write(compressed)

    print(f"\n--- Output ---")
    print(f"  Raw protobuf:    {bin_path} ({len(raw)} bytes)")
    print(f"  Gzipped:         {gz_path} ({len(compressed)} bytes)")
    print(f"  Batch UUID:      {batch_uuid}")
    print(f"  S3 key would be: ingest/v1/partner/{batch_uuid}/{batch_uuid}.gz")

    b64 = base64.b64encode(raw).decode()
    print(f"\n--- Base64 (raw protobuf) ---")
    if len(b64) > 2000:
        print(f"  {b64[:2000]}... (truncated, {len(b64)} chars total)")
    else:
        print(f"  {b64}")

    return bin_path, gz_path


def print_summary(batch):
    """Print human-readable summary of the batch contents."""
    print(f"\n--- Batch summary ---")
    print(f"  created_at: "
          f"{batch.created_at.ToDatetime(tzinfo=timezone.utc).isoformat()}")
    print(f"  total records: {len(batch.records)}")
    print()
    for i, rec in enumerate(batch.records):
        if i >= 20:
            print(f"  ... and {len(batch.records) - 20} more records")
            break
        labels_str = {ns: list(nl.slugs) for ns, nl in rec.labels.items()}
        print(f"  [{i}] {rec.entity_drn_id}  ->  {labels_str}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert Snowflake dietary tags to labels-store protobuf",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="See script docstring for full documentation.",
    )
    parser.add_argument("--threshold", type=float, default=50.0,
                        help="Min GMV %% to qualify for a label (default: 50)")
    parser.add_argument("--namespace", type=str, default="dietary-webster",
                        help="Protobuf label namespace (default: dietary-webster)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of stores fetched (for testing)")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Directory for output files (default: .)")

    sf = parser.add_argument_group("Snowflake connection",
        "All values default to ~/.snowflake/connections.toml if present.")
    sf.add_argument("--sf-account", type=str, default=None)
    sf.add_argument("--sf-user", type=str, default=None)
    sf.add_argument("--sf-warehouse", type=str, default=None)
    sf.add_argument("--sf-database", type=str, default=None)
    sf.add_argument("--sf-schema", type=str, default=None)
    sf.add_argument("--sf-token", type= "**********"=None,
                    help= "**********"

    args = parser.parse_args()

    # Merge CLI args with connections.toml defaults
    toml_cfg = _read_connections_toml()
    account   = args.sf_account   or toml_cfg.get("account", "DOORDASH")
    user      = args.sf_user      or toml_cfg.get("user")
    warehouse = args.sf_warehouse or toml_cfg.get("warehouse", "ADHOC")
    database  = args.sf_database  or toml_cfg.get("database", "PRODDB")
    schema    = args.sf_schema    or toml_cfg.get("schema", "PUBLIC")
    token     = "**********"

    if not user:
        parser.error("Snowflake user is required. Set --sf-user or configure "
                     "~/.snowflake/connections.toml")

 "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********": "**********"
        print(f"Using token from {'CLI' if args.sf_token else '~/.snowflake/connections.toml'}")

    # 1. Connect
    conn = "**********"

    try:
        # 2. Create temp tables
        print("\nCreating temp tables ...")
        run_setup(conn)

        # 3. Fetch store stats
        print(f"\nFetching store stats (limit={args.limit}) ...")
        rows = fetch_store_stats(conn, limit=args.limit)
        print(f"  Fetched {len(rows)} stores.")

        if not rows:
            print("No data returned. Exiting.")
            return

        # 4. Build protobuf
        print(f"\nBuilding protobuf "
              f"(threshold={args.threshold}%, namespace={args.namespace}) ...")
        batch = build_batch(rows, args.threshold, args.namespace)

        if len(batch.records) == 0:
            print("No stores met the threshold. Nothing to write.")
            return

        # 5. Write output
        serialize_and_write(batch, args.output_dir)

        # 6. Summary
        print_summary(batch)

    finally:
        conn.close()
        print("\nDone.")


if __name__ == "__main__":
    main()
