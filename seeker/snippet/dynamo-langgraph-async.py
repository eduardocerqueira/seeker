#date: 2025-04-16T17:04:36Z
#url: https://api.github.com/gists/8fc764a847578b67a88e1ad16dc26a0b
#owner: https://api.github.com/users/STHITAPRAJNAS

import asyncio
import pickle
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional, Sequence, Tuple, Dict, Any, List, Union

import aiobotocore.session
from botocore.exceptions import ClientError
from boto3.dynamodb.types import TypeDeserializer, TypeSerializer
from langgraph.serde.base import BaseSerializer
from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint, CheckpointMetadata, CheckpointTuple

# --- Constants for DynamoDB Table Structure (matching the original library) ---
DEFAULT_TABLE_NAME = "langgraph_checkpoints"
DEFAULT_PK = "thread_id"
DEFAULT_SK_CHECKPOINT = "checkpoint" # Sort key for the main checkpoint data
DEFAULT_SK_METADATA_PREFIX = "metadata|" # Prefix for metadata sort keys
DEFAULT_TTL_KEY = "ttl_timestamp" # Optional TTL attribute

class PickleSerializer(BaseSerializer):
    """Serializer that uses pickle."""
    def dumps(self, obj: Any) -> bytes:
        return pickle.dumps(obj)

    def loads(self, data: bytes) -> Any:
        return pickle.loads(data)

class AsyncDynamoDBSaver(BaseCheckpointSaver):
    """
    An asynchronous checkpoint saver that stores checkpoints in DynamoDB.

    Args:
        table_name (str): The name of the DynamoDB table. Defaults to "langgraph_checkpoints".
        primary_key (str): The name of the partition key column. Defaults to "thread_id".
        sort_key_checkpoint (str): The sort key value for the main checkpoint data. Defaults to "checkpoint".
        sort_key_metadata_prefix (str): The prefix for sort keys storing metadata. Defaults to "metadata|".
        ttl_key (Optional[str]): The name of the attribute to use for TTL. If None, TTL is not used. Defaults to "ttl_timestamp".
        serializer (Optional[BaseSerializer]): The serializer to use for checkpoint data. Defaults to PickleSerializer.
        aws_region (Optional[str]): AWS region name. If None, uses default from environment/config.
        aws_access_key_id (Optional[str]): "**********"
        aws_secret_access_key (Optional[str]): "**********"
        endpoint_url (Optional[str]): Custom DynamoDB endpoint URL (e.g., for DynamoDB Local).
    """

    def __init__(
        self,
        *,
        table_name: str = DEFAULT_TABLE_NAME,
        primary_key: str = DEFAULT_PK,
        sort_key_checkpoint: str = DEFAULT_SK_CHECKPOINT,
        sort_key_metadata_prefix: str = DEFAULT_SK_METADATA_PREFIX,
        ttl_key: Optional[str] = DEFAULT_TTL_KEY,
        serializer: Optional[BaseSerializer] = None,
        aws_region: Optional[str] = None,
        aws_access_key_id: "**********"
        aws_secret_access_key: "**********"
        endpoint_url: Optional[str] = None,
    ):
        super().__init__(serializer=serializer or PickleSerializer())
        self.table_name = table_name
        self.primary_key = primary_key
        self.sort_key_checkpoint = sort_key_checkpoint
        self.sort_key_metadata_prefix = sort_key_metadata_prefix
        self.ttl_key = ttl_key
        self.serializer = self.serde

        # --- aiobotocore Session and Client Setup ---
        self.session = aiobotocore.session.get_session()
        self.client_config = {
            "service_name": "dynamodb",
            "region_name": aws_region,
            "aws_access_key_id": "**********"
            "aws_secret_access_key": "**********"
            "endpoint_url": endpoint_url,
        }

        # --- DynamoDB Type Serializer/Deserializer ---
        # These are synchronous but used within async methods for data transformation
        self._type_serializer = TypeSerializer()
        self._type_deserializer = TypeDeserializer()

    @asynccontextmanager
    async def _get_client(self) -> AsyncGenerator[Any, None]:
        """Provides an asynchronous DynamoDB client context."""
        async with self.session.create_client(**self.client_config) as client:
            yield client

    def _serialize_item(self, item: Dict[str, Any]) -> Dict[str, Dict]:
        """Serializes a Python dict into DynamoDB attribute value format."""
        return {k: self._type_serializer.serialize(v) for k, v in item.items()}

    def _deserialize_item(self, item: Dict[str, Dict]) -> Dict[str, Any]:
        """Deserializes a DynamoDB item into a Python dict."""
        return {k: self._type_deserializer.deserialize(v) for k, v in item.items()}

    async def aget_tuple(self, config: Dict[str, Any]) -> Optional[CheckpointTuple]:
        """
        Asynchronously retrieves a checkpoint tuple (checkpoint, metadata, parent_config)
        for a given thread configuration.

        Args:
            config: The configuration identifying the thread (must contain 'thread_id').

        Returns:
            An optional CheckpointTuple if found, otherwise None.
        """
        thread_id = config["thread_id"]
        async with self._get_client() as client:
            try:
                # Query for both checkpoint and metadata items for the thread_id
                response = await client.query(
                    TableName=self.table_name,
                    KeyConditionExpression=f"{self.primary_key} = :pk",
                    ExpressionAttributeValues={
                        ":pk": self._type_serializer.serialize(thread_id),
                    },
                    ConsistentRead=True, # Ensure we get the latest data
                )
            except ClientError as e:
                # Handle potential errors like table not found
                print(f"Error querying DynamoDB: {e}")
                # Consider more specific error handling (e.g., ResourceNotFoundException)
                return None

            items = [self._deserialize_item(item) for item in response.get("Items", [])]

            checkpoint_item = None
            metadata_items = []

            # Separate checkpoint and metadata items
            for item in items:
                if item.get("sk") == self.sort_key_checkpoint:
                    checkpoint_item = item
                elif isinstance(item.get("sk"), str) and item["sk"].startswith(self.sort_key_metadata_prefix):
                    metadata_items.append(item)

            if not checkpoint_item:
                return None # No checkpoint found for this thread_id

            # Deserialize checkpoint data
            try:
                checkpoint_data = checkpoint_item.get("checkpoint")
                if isinstance(checkpoint_data, bytes):
                    checkpoint = self.serializer.loads(checkpoint_data)
                elif checkpoint_data is None: # Handle case where checkpoint is empty
                     checkpoint = Checkpoint(v=1, ts="", id="", channel_values={}, channel_versions={}, seen_recovery_events=set(), config={})
                else:
                    # Should ideally be bytes, handle unexpected types if necessary
                    print(f"Warning: Unexpected checkpoint data type: {type(checkpoint_data)}")
                    return None # Or raise an error
            except (pickle.UnpicklingError, TypeError, EOFError) as e:
                print(f"Error deserializing checkpoint data for thread {thread_id}: {e}")
                return None # Corrupted data

            # Construct metadata dictionary
            metadata = {}
            for meta_item in metadata_items:
                # Extract ts from the sort key (e.g., "metadata|2024-01-01T12:00:00Z")
                try:
                    ts = meta_item["sk"].split(self.sort_key_metadata_prefix, 1)[1]
                    meta_data_bytes = meta_item.get("metadata")
                    if isinstance(meta_data_bytes, bytes):
                         # Deserialize metadata if it's stored (adjust if your schema differs)
                         # Assuming metadata is stored as serialized bytes similar to checkpoint
                         meta_dict = self.serializer.loads(meta_data_bytes)
                         # Ensure it's a dictionary, default to empty if deserialization fails or type is wrong
                         metadata[ts] = meta_dict if isinstance(meta_dict, dict) else {}
                    else:
                         # If metadata is not stored as bytes or is missing, use an empty dict
                         metadata[ts] = {}
                except (IndexError, pickle.UnpicklingError, TypeError, EOFError) as e:
                    print(f"Warning: Could not process metadata item {meta_item.get('sk')} for thread {thread_id}: {e}")
                    # Assign empty dict for this potentially corrupted/malformed metadata entry
                    ts_key = meta_item.get('sk', 'unknown').split(self.sort_key_metadata_prefix, 1)[-1]
                    if ts_key != 'unknown':
                        metadata[ts_key] = {}


            # Find the parent config (config of the checkpoint one step before the current one)
            parent_config = None
            if checkpoint and checkpoint.id: # Check if checkpoint and its ID exist
                sorted_metadata_ts = sorted(metadata.keys(), reverse=True)
                # Find the timestamp corresponding to the current checkpoint's ID
                try:
                    current_checkpoint_idx = sorted_metadata_ts.index(checkpoint.ts)
                    # The parent is the next one in the sorted list (if it exists)
                    if current_checkpoint_idx + 1 < len(sorted_metadata_ts):
                        parent_ts = sorted_metadata_ts[current_checkpoint_idx + 1]
                        # Assuming metadata contains the 'config' needed
                        parent_config = metadata.get(parent_ts, {}).get("config")
                except ValueError:
                     # This might happen if the checkpoint.ts is not found in metadata keys
                     print(f"Warning: Checkpoint timestamp {checkpoint.ts} not found in metadata for thread {thread_id}.")
                     # Attempt to find the latest metadata entry as a fallback parent if logical
                     if sorted_metadata_ts:
                         latest_meta_ts = sorted_metadata_ts[0]
                         # Check if the latest metadata isn't the current checkpoint itself
                         if latest_meta_ts != checkpoint.ts:
                              parent_config = metadata.get(latest_meta_ts, {}).get("config")


            return CheckpointTuple(config=config, checkpoint=checkpoint, metadata=metadata, parent_config=parent_config)

    async def alist(
        self,
        config: Optional[Dict[str, Any]],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> AsyncGenerator[CheckpointTuple, None]:
        """
        Asynchronously lists checkpoints for a given thread configuration,
        optionally filtered and limited.

        Args:
            config: The configuration identifying the thread (must contain 'thread_id').
            filter: Optional dictionary for filtering based on metadata attributes.
                    (Note: DynamoDB filtering capabilities might be limited without secondary indexes).
            before: Optional config to list checkpoints strictly before the one specified by this config's timestamp.
            limit: The maximum number of checkpoints to return.

        Yields:
            CheckpointTuple objects matching the criteria.
        """
        if config is None:
             # Scan is generally discouraged for performance reasons on large tables.
             # If listing across all threads is essential, consider if a GSI is feasible.
             # This implementation will perform a scan if no thread_id is provided.
             print("Warning: Listing checkpoints without a thread_id will perform a DynamoDB Scan, which can be inefficient and costly on large tables.")
             yield await self._scan_all_checkpoints(filter=filter, limit=limit) # Delegate to a scan helper
             return

        thread_id = config.get("thread_id")
        if not thread_id:
            raise ValueError("Configuration must include 'thread_id' to list checkpoints.")

        async with self._get_client() as client:
            query_kwargs = {
                "TableName": self.table_name,
                "KeyConditionExpression": f"{self.primary_key} = :pk AND begins_with(sk, :sk_prefix)",
                "ExpressionAttributeValues": {
                    ":pk": self._type_serializer.serialize(thread_id),
                    ":sk_prefix": self._type_serializer.serialize(self.sort_key_metadata_prefix),
                },
                "ScanIndexForward": False, # List newest first by timestamp in sort key
                "ConsistentRead": True,
            }

            # --- Filtering Logic ---
            # Basic 'before' filtering based on timestamp in the sort key
            if before and before.get("thread_id") == thread_id:
                 # We need the timestamp associated with the 'before' config.
                 # This requires getting the checkpoint tuple for 'before' first.
                 before_tuple = await self.aget_tuple(before)
                 if before_tuple and before_tuple.checkpoint:
                     before_ts = before_tuple.checkpoint.ts
                     # Modify KeyConditionExpression to fetch items *older* than 'before_ts'
                     # DynamoDB sorts lexicographically, so '<' works for ISO timestamps
                     query_kwargs["KeyConditionExpression"] = f"{self.primary_key} = :pk AND sk < :sk_before"
                     query_kwargs["ExpressionAttributeValues"][":sk_before"] = self._type_serializer.serialize(f"{self.sort_key_metadata_prefix}{before_ts}")
                 else:
                      # If 'before' config doesn't resolve to a checkpoint, return nothing
                      print(f"Warning: 'before' config {before} did not resolve to a valid checkpoint. Returning no results.")
                      return


            # Advanced filtering based on 'filter' dict (requires potentially complex FilterExpressions)
            # Note: Filtering happens *after* the query retrieves items, making it less efficient
            # than filtering on indexed attributes. This implementation provides basic support.
            filter_expression_parts = []
            if filter:
                 # This part is complex because metadata is serialized within the 'metadata' attribute.
                 # Direct filtering on nested attributes inside the serialized blob isn't possible
                 # without scanning or potentially structuring metadata differently (e.g., separate attributes).
                 # We will filter *after* retrieving the items for simplicity here.
                 print("Warning: 'filter' parameter currently filters results *after* retrieval from DynamoDB due to serialized metadata. This might be inefficient. Consider optimizing schema or using Scan for complex filters if needed.")
                 pass # Filtering logic will be applied post-retrieval


            # --- Pagination and Limit ---
            paginator = client.get_paginator("query")
            pages = paginator.paginate(**query_kwargs)

            count = 0
            async for page in pages:
                items = [self._deserialize_item(item) for item in page.get("Items", [])]
                # Sort by timestamp descending (lexicographically on sort key)
                items.sort(key=lambda x: x.get("sk", ""), reverse=True)

                for meta_item in items:
                    if limit is not None and count >= limit:
                        return # Stop yielding if limit is reached

                    # Extract timestamp and attempt to load metadata
                    try:
                        ts = meta_item["sk"].split(self.sort_key_metadata_prefix, 1)[1]
                        meta_data_bytes = meta_item.get("metadata")
                        if isinstance(meta_data_bytes, bytes):
                            metadata_entry = self.serializer.loads(meta_data_bytes)
                            if not isinstance(metadata_entry, dict): metadata_entry = {} # Ensure dict
                        else:
                            metadata_entry = {} # Default to empty dict if no metadata stored

                        # Construct the config for this specific checkpoint timestamp
                        checkpoint_specific_config = {**config, "checkpoint_ts": ts}

                        # --- Apply Post-Retrieval Filter ---
                        if filter:
                            match = True
                            for key, value in filter.items():
                                # Check if the key exists in the *deserialized* metadata and matches the value
                                if metadata_entry.get(key) != value:
                                    match = False
                                    break
                            if not match:
                                continue # Skip this item if it doesn't match the filter

                        # --- Get the full checkpoint tuple for this timestamp ---
                        # This involves another call to aget_tuple, which might be inefficient.
                        # A potential optimization is to fetch the main checkpoint item
                        # along with metadata in the initial query if possible, or reconstruct
                        # the CheckpointTuple more directly if the main checkpoint data isn't needed
                        # for the list operation itself (depends on use case).
                        # For now, we re-fetch for consistency.
                        # We need to reconstruct the config that *would* lead to this checkpoint
                        # This is tricky. The 'list' operation usually returns configs that *can* be used
                        # to fetch a specific state. Let's yield the config associated with the metadata entry.
                        # The caller can then use this config in `aget_tuple`.

                        # Construct a config representing this specific point in time
                        list_item_config = metadata_entry.get("config", config) # Use stored config if available

                        # Fetch the full tuple using the derived config/timestamp
                        # We need the *actual* checkpoint data corresponding to this metadata entry's timestamp
                        # This requires fetching the main checkpoint item *again* based on thread_id
                        # and then finding the specific version matching 'ts'. This is inefficient.
                        # A better approach for `list` might be to yield only metadata or configs,
                        # or to optimize the query/data structure.

                        # Let's fetch the *latest* checkpoint tuple for the thread_id
                        # and then find the relevant entry in its metadata. This avoids N+1 fetches inside the loop.
                        # This assumes `alist` is primarily used to find *configs* to then load.

                        # Yielding the config derived from metadata seems most aligned with BaseCheckpointSaver.alist
                        yield CheckpointTuple(config=list_item_config, checkpoint=None, metadata=metadata_entry, parent_config=None) # Yielding config/metadata only

                        count += 1

                    except (IndexError, pickle.UnpicklingError, TypeError, EOFError) as e:
                        print(f"Warning: Could not process metadata item {meta_item.get('sk')} during list for thread {thread_id}: {e}")
                        continue # Skip corrupted/malformed items

                if limit is not None and count >= limit:
                    break # Exit pagination early if limit reached

    async def _scan_all_checkpoints(
        self,
        *,
        filter: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None
    ) -> AsyncGenerator[CheckpointTuple, None]:
        """ Helper to scan the entire table. Use with caution. """
        async with self._get_client() as client:
            paginator = client.get_paginator("scan")
            scan_kwargs = {"TableName": self.table_name}

            # Scan retrieves *all* items, then filters. We need to filter for checkpoint items.
            # We'll filter client-side after retrieving pages.

            count = 0
            seen_thread_ids = set()
            async for page in paginator.paginate(**scan_kwargs):
                items = [self._deserialize_item(item) for item in page.get("Items", [])]

                # Process items to find unique thread_ids and their latest checkpoint/metadata
                # This is complex because scan gives us everything mixed together.
                # We are aiming to yield one CheckpointTuple per thread_id found.

                thread_checkpoints = {} # {thread_id: latest_checkpoint_item}
                thread_metadata = {} # {thread_id: {ts: metadata_dict}}

                for item in items:
                    thread_id = item.get(self.primary_key)
                    sort_key = item.get("sk")
                    if not thread_id or not sort_key:
                        continue # Skip items missing primary or sort key

                    if thread_id not in thread_metadata:
                        thread_metadata[thread_id] = {}

                    if sort_key == self.sort_key_checkpoint:
                         # Keep track of the main checkpoint item for the thread
                         # (Assuming only one such item per thread_id based on original schema)
                         thread_checkpoints[thread_id] = item
                    elif isinstance(sort_key, str) and sort_key.startswith(self.sort_key_metadata_prefix):
                        try:
                            ts = sort_key.split(self.sort_key_metadata_prefix, 1)[1]
                            meta_data_bytes = item.get("metadata")
                            if isinstance(meta_data_bytes, bytes):
                                metadata_entry = self.serializer.loads(meta_data_bytes)
                                if not isinstance(metadata_entry, dict): metadata_entry = {}
                            else:
                                metadata_entry = {}
                            thread_metadata[thread_id][ts] = metadata_entry
                        except Exception as e:
                             print(f"Warning: Could not process metadata item {sort_key} during scan for thread {thread_id}: {e}")


                # Now, for each thread found in this page, construct and yield its latest CheckpointTuple
                for thread_id, checkpoint_item in thread_checkpoints.items():
                     if thread_id in seen_thread_ids:
                         continue # Avoid yielding the same thread again if paginated

                     if limit is not None and count >= limit:
                         return # Stop if limit reached

                     metadata_for_thread = thread_metadata.get(thread_id, {})

                     try:
                         checkpoint_data = checkpoint_item.get("checkpoint")
                         if isinstance(checkpoint_data, bytes):
                             checkpoint = self.serializer.loads(checkpoint_data)
                         elif checkpoint_data is None:
                             checkpoint = Checkpoint(v=1, ts="", id="", channel_values={}, channel_versions={}, seen_recovery_events=set(), config={})
                         else:
                             continue # Skip if checkpoint data is invalid type
                     except Exception as e:
                         print(f"Error deserializing checkpoint data during scan for thread {thread_id}: {e}")
                         continue # Skip this thread if checkpoint is corrupted

                     # --- Apply Post-Retrieval Filter (on the latest checkpoint's metadata) ---
                     latest_ts = checkpoint.ts
                     latest_metadata_entry = metadata_for_thread.get(latest_ts, {})
                     if filter:
                         match = True
                         for key, value in filter.items():
                             if latest_metadata_entry.get(key) != value:
                                 match = False
                                 break
                         if not match:
                             continue # Skip this thread if it doesn't match the filter

                     # Determine parent config (similar logic as in aget_tuple)
                     parent_config = None
                     sorted_metadata_ts = sorted(metadata_for_thread.keys(), reverse=True)
                     try:
                         current_checkpoint_idx = sorted_metadata_ts.index(latest_ts)
                         if current_checkpoint_idx + 1 < len(sorted_metadata_ts):
                             parent_ts = sorted_metadata_ts[current_checkpoint_idx + 1]
                             parent_config = metadata_for_thread.get(parent_ts, {}).get("config")
                     except ValueError:
                         pass # Checkpoint ts might not be in metadata keys

                     config = checkpoint.config # Use the config stored within the latest checkpoint

                     yield CheckpointTuple(
                         config=config,
                         checkpoint=checkpoint,
                         metadata=metadata_for_thread, # Yield all metadata for the thread
                         parent_config=parent_config
                     )
                     seen_thread_ids.add(thread_id)
                     count += 1

                     if limit is not None and count >= limit:
                         return # Check limit again after yielding


    async def aput(
        self, config: Dict[str, Any], checkpoint: Checkpoint, metadata: CheckpointMetadata
    ) -> Dict[str, Any]:
        """
        Asynchronously saves a checkpoint and its metadata to DynamoDB.

        Args:
            config: The configuration identifying the thread (must contain 'thread_id').
            checkpoint: The checkpoint object to save.
            metadata: The metadata associated with the checkpoint.

        Returns:
            The original config dictionary.
        """
        thread_id = config["thread_id"]
        serialized_checkpoint = self.serializer.dumps(checkpoint)
        serialized_metadata = self.serializer.dumps(metadata) # Serialize the whole metadata dict

        async with self._get_client() as client:
            # Prepare the main checkpoint item
            checkpoint_item = {
                self.primary_key: thread_id,
                "sk": self.sort_key_checkpoint, # Static sort key for the main checkpoint data
                "checkpoint": serialized_checkpoint,
                "ts": checkpoint.ts, # Store timestamp for potential queries/sorting if needed
                # Add other top-level fields from checkpoint if desired (e.g., 'v')
            }
            # Optionally add TTL
            if self.ttl_key and checkpoint.ts:
                 # Assuming checkpoint.ts is an ISO 8601 string
                 # Convert to Unix timestamp for DynamoDB TTL
                 try:
                     from datetime import datetime, timezone
                     # Example: Add TTL of 30 days from checkpoint timestamp
                     # Adjust the duration as needed
                     dt_obj = datetime.fromisoformat(checkpoint.ts.replace("Z", "+00:00"))
                     # You might want to add a fixed TTL duration instead of basing it on ts
                     # ttl_timestamp = int((dt_obj + timedelta(days=30)).timestamp())
                     # For simplicity, let's just use the checkpoint timestamp directly if TTL is based on it
                     # ttl_timestamp = int(dt_obj.timestamp())
                     # Or more commonly, set TTL based on *write time* + duration
                     # ttl_timestamp = int(time.time()) + (30 * 24 * 60 * 60) # 30 days from now
                     # Let's assume the user wants TTL based on the checkpoint's timestamp + some offset
                     # If checkpoint.ts is the *expiry* time, use it directly.
                     # If it's the creation time, add an offset.
                     # For this example, let's assume ts IS the expiry timestamp (needs clarification based on use case)
                     # ttl_timestamp = int(dt_obj.timestamp())
                     # checkpoint_item[self.ttl_key] = ttl_timestamp
                     pass # TTL logic needs refinement based on exact requirement
                 except Exception as e:
                     print(f"Warning: Could not calculate TTL for thread {thread_id}: {e}")


            # Prepare the metadata item
            metadata_item = {
                self.primary_key: thread_id,
                "sk": f"{self.sort_key_metadata_prefix}{checkpoint.ts}", # Sort key includes timestamp
                "metadata": serialized_metadata, # Store the serialized metadata blob
                # Optionally add TTL to metadata items as well
            }
            # if self.ttl_key and checkpoint.ts and self.ttl_key in checkpoint_item:
            #     metadata_item[self.ttl_key] = checkpoint_item[self.ttl_key]


            # Use BatchWriteItem for atomicity (or TransactWriteItems for stronger guarantees if needed)
            # For simplicity, we use two separate PutItem calls here.
            # Consider BatchWriteItem or TransactWriteItems for production if atomicity is critical.
            try:
                # Put the main checkpoint item
                await client.put_item(
                    TableName=self.table_name,
                    Item=self._serialize_item(checkpoint_item),
                )
                # Put the metadata item
                await client.put_item(
                    TableName=self.table_name,
                    Item=self._serialize_item(metadata_item),
                )
            except ClientError as e:
                print(f"Error putting checkpoint/metadata for thread {thread_id}: {e}")
                # Re-raise or handle specific errors (e.g., ValidationException)
                raise

        return config

# Example Usage (requires async environment)
async def main():
    # Configure the saver (use endpoint_url for DynamoDB Local)
    # saver = AsyncDynamoDBSaver(endpoint_url="http://localhost:8000")
    saver = AsyncDynamoDBSaver() # Assumes AWS credentials and region are configured

    # Example config and checkpoint data
    thread_config = {"thread_id": "thread-124-async"}
    initial_checkpoint = Checkpoint(
        v=1,
        ts="2025-04-16T10:00:00Z",
        id="checkpoint_0",
        channel_values={"messages": ["hello from async"]},
        channel_versions={"messages": 1},
        seen_recovery_events=set(),
        config=thread_config,
    )
    initial_metadata = {
        "source": "user",
        "step": 1,
        "writes": {"chatbot": {"messages": ["hello from async"]}},
        "config": thread_config, # Include config in metadata
    }

    # --- Put Checkpoint ---
    print(f"Putting checkpoint for: {thread_config['thread_id']}")
    await saver.aput(thread_config, initial_checkpoint, initial_metadata)
    print("Put successful.")

    # --- Get Checkpoint ---
    print(f"\nGetting checkpoint tuple for: {thread_config['thread_id']}")
    retrieved_tuple = await saver.aget_tuple(thread_config)
    if retrieved_tuple:
        print("Retrieved Checkpoint:", retrieved_tuple.checkpoint)
        print("Retrieved Metadata:", retrieved_tuple.metadata)
        print("Retrieved Parent Config:", retrieved_tuple.parent_config)
    else:
        print("Checkpoint not found.")

    # --- Add another checkpoint ---
    next_checkpoint = Checkpoint(
        v=1,
        ts="2025-04-16T10:05:00Z",
        id="checkpoint_1",
        channel_values={"messages": ["hello from async", "how are you?"]},
        channel_versions={"messages": 2},
        seen_recovery_events=set(),
        config=thread_config, # Config associated with *this* state
    )
    next_metadata = {
        "source": "llm",
        "step": 2,
        "writes": {"chatbot": {"messages": ["how are you?"]}},
        "config": thread_config,
    }
    print(f"\nPutting next checkpoint for: {thread_config['thread_id']}")
    await saver.aput(thread_config, next_checkpoint, next_metadata)
    print("Put successful.")

    # --- Get Updated Checkpoint ---
    print(f"\nGetting updated checkpoint tuple for: {thread_config['thread_id']}")
    retrieved_tuple_updated = await saver.aget_tuple(thread_config)
    if retrieved_tuple_updated:
        print("Updated Checkpoint:", retrieved_tuple_updated.checkpoint)
        # Metadata should now contain both timestamps
        print("Updated Metadata:", retrieved_tuple_updated.metadata)
        # Parent config should point to the config of the previous checkpoint
        print("Updated Parent Config:", retrieved_tuple_updated.parent_config)
        assert retrieved_tuple_updated.parent_config == thread_config # In this simple case
    else:
        print("Updated Checkpoint not found.")


    # --- List Checkpoints ---
    print(f"\nListing checkpoints for: {thread_config['thread_id']}")
    async for checkpoint_tuple in saver.alist(thread_config, limit=5):
         # Note: The yielded tuple from alist in this impl contains config/metadata
         # You'd typically use the config to fetch the full checkpoint if needed
         print(f"  - Config: {checkpoint_tuple.config}, Metadata TS: {checkpoint_tuple.config.get('checkpoint_ts')}, Metadata Source: {checkpoint_tuple.metadata.get('source')}")

    # --- List Checkpoints Before a specific one ---
    # Config representing the state *at* the second checkpoint
    config_at_second_checkpoint = {**thread_config, "checkpoint_ts": "2025-04-16T10:05:00Z"}
    print(f"\nListing checkpoints before {config_at_second_checkpoint['checkpoint_ts']}:")
    async for checkpoint_tuple in saver.alist(thread_config, before=config_at_second_checkpoint):
         print(f"  - Config: {checkpoint_tuple.config}, Metadata TS: {checkpoint_tuple.config.get('checkpoint_ts')}, Metadata Source: {checkpoint_tuple.metadata.get('source')}")
         # Should only list the first checkpoint ("2025-04-16T10:00:00Z")

    # --- List Checkpoints with Filter (Example - may be inefficient) ---
    print(f"\nListing checkpoints with source='user':")
    async for checkpoint_tuple in saver.alist(thread_config, filter={"source": "user"}):
         print(f"  - Config: {checkpoint_tuple.config}, Metadata TS: {checkpoint_tuple.config.get('checkpoint_ts')}, Metadata Source: {checkpoint_tuple.metadata.get('source')}")


if __name__ == "__main__":
    # Ensure you have a running DynamoDB instance (local or AWS)
    # and necessary credentials/region configured.
    # You might need to create the table first:
    # aws dynamodb create-table \
    #    --table-name langgraph_checkpoints \
    #    --attribute-definitions AttributeName=thread_id,AttributeType=S AttributeName=sk,AttributeType=S \
    #    --key-schema AttributeName=thread_id,KeyType=HASH AttributeName=sk,KeyType=RANGE \
    #    --billing-mode PAY_PER_REQUEST \
    #    --endpoint-url http://localhost:8000 # Optional: for DynamoDB Local
    #
    # To enable TTL:
    # aws dynamodb update-time-to-live --table-name langgraph_checkpoints \
    #    --time-to-live-specification "Enabled=true, AttributeName=ttl_timestamp" \
    #    --endpoint-url http://localhost:8000 # Optional
    try:
        asyncio.run(main())
    except ImportError:
        print("Please install necessary libraries: pip install aiobotocore langgraph boto3")
    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceNotFoundException':
            print("\nError: DynamoDB table 'langgraph_checkpoints' not found.")
            print("Please create the table using the AWS CLI command provided in the script comments.")
        elif 'Credentials' in str(e):
             print(f"\nError: AWS Credentials not found or invalid: {e}")
             print("Ensure your AWS credentials (access key, secret key, region) are configured correctly (e.g., via environment variables, ~/.aws/credentials).")
        else:
            print(f"\nAn AWS ClientError occurred: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")


        
        
        
### Example Implementatation langgraph

import asyncio
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.base import Checkpoint
# Assume AsyncDynamoDBSaver class is defined as in the document
# from your_module import AsyncDynamoDBSaver

# 1. Define your graph state
class AgentState(TypedDict):
    input: str
    output: str

# 2. Define your graph nodes (ensure they are async if doing async work)
async def node_a(state: AgentState):
    print("---Executing Node A---")
    # Simulate async work
    await asyncio.sleep(0.1)
    return {"output": f"Output from A based on '{state['input']}'"}

async def node_b(state: AgentState):
    print("---Executing Node B---")
    await asyncio.sleep(0.1)
    return {"output": state['output'] + " | Output from B"}

# 3. Instantiate the Async Checkpointer
# Configure with your table name, region, etc.
# Use endpoint_url for DynamoDB Local if needed
checkpointer = AsyncDynamoDBSaver(
    table_name="langgraph_checkpoints",
    # aws_region="your-region", # Optional
    # endpoint_url="http://localhost:8000" # Optional
)

# 4. Build the graph, passing the checkpointer
builder = StateGraph(AgentState)
builder.add_node("a", node_a)
builder.add_node("b", node_b)
builder.add_edge(START, "a")
builder.add_edge("a", "b")
builder.add_edge("b", END)

# Pass the single async checkpointer instance here
graph = builder.compile(checkpointer=checkpointer)

# 5. Use the graph asynchronously
async def run_graph():
    thread_config = {"configurable": {"thread_id": "my-thread-async-01"}}
    print("Running graph...")
    async for event in graph.astream_events(
        {"input": "hello async world"}, config=thread_config, version="v1"
    ):
         kind = event["event"]
         if kind == "on_chain_end":
             print(f"---Graph Ended with Output---")
             print(event["data"]["output"])
         elif kind == "on_checkpoint":
             print("---Checkpoint Saved---")
             # print(event["data"]) # Can inspect checkpoint data

    # You can retrieve the checkpoint later using the same checkpointer instance
    print("\nRetrieving final checkpoint:")
    final_checkpoint_tuple = await checkpointer.aget_tuple(thread_config["configurable"])
    if final_checkpoint_tuple:
        print(final_checkpoint_tuple.checkpoint.channel_values) # Print final state

# Run the async function
# asyncio.run(run_graph())_tuple:
        print(final_checkpoint_tuple.checkpoint.channel_values) # Print final state

# Run the async function
# asyncio.run(run_graph())