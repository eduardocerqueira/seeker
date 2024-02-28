#date: 2024-02-28T16:47:24Z
#url: https://api.github.com/gists/9f2c8c618b00aa1c599b7d9fc9d993fb
#owner: https://api.github.com/users/rudolfix

from __future__ import annotations
from collections import defaultdict
from typing import List, cast

import airbyte as ab
from airbyte import exceptions as exc
from airbyte.strategies import WriteStrategy
from airbyte_protocol.models import (
    AirbyteRecordMessage, AirbyteStateMessage, AirbyteStateType, AirbyteStreamState, Type as MessageType,
    DestinationSyncMode, SyncMode)
from airbyte._util import protocol_util
from airbyte.telemetry import CacheTelemetryInfo

import dlt
from dlt.common.schema.typing import TWriteDisposition
from dlt.common.normalizers.naming import snake_case

dlt_naming = snake_case.NamingConvention()


def convert_to_dlt_source(
    ab_source: ab.Source,
    streams: List[str],
    force_full_refresh: bool = False,
):
    """This function converts Airbyte source into dlt source
    
       1. streams are mapped in resources
       2. state is handled by dlt
       3. we support primary keys, merge, append and replace loads
       4. we support full refresh

       Notes:
       1. we do not port catalogue schema into dlt schema. should be trivial but IMO it is better
          to rely on dlt schema inference.
       2. Note that AirbytePy is not doing that too (as far as I know)
       3. You can attach any transforms, pydantic models or whatever you care to the resources

       WARNING: tested on poke api only
    """
    if streams:
        ab_source.select_streams(streams)

    if not ab_source._selected_stream_names:
        raise exc.AirbyteLibNoStreamsSelectedError(
            connector_name=ab_source.name,
            available_streams=ab_source.get_available_streams(),
        )
    
    def process_airbyte_messages(
        force_full_refresh: bool,
        *,
        max_batch_size: int = 20,
    ):
        # use dlt state as airbyte state
        state = dlt.current.source_state()
        if force_full_refresh:
            state.clear()
        state_messages = [AirbyteStateMessage.parse_obj(s) for s in state.values()]
        
        stream_batches: dict[str, list[dict]] = defaultdict(list, {})
        # Process messages, writing to batches as we go
        for message in ab_source._read(CacheTelemetryInfo("guess_who"), state_messages):
            if message.type is MessageType.RECORD:
                record_msg = cast(AirbyteRecordMessage, message.record)
                stream_name = record_msg.stream
                stream_batch = stream_batches[stream_name]
                stream_batch.append(protocol_util.airbyte_record_message_to_dict(record_msg))
                if len(stream_batch) >= max_batch_size:
                    yield {"stream_name": stream_name, "batch": stream_batch}
                    # allow for new batch while the old one is still processing
                    stream_batches[stream_name] = {}

            elif message.type is MessageType.STATE:
                state_msg = cast(AirbyteStateMessage, message.state)
                if state_msg.type in [AirbyteStateType.GLOBAL, AirbyteStateType.LEGACY]:
                    # interestingly Airbyte lib does not handle finalization of this state ;>
                    state_key = f"_{state_msg.type}"
                else:
                    stream_state = cast(AirbyteStreamState, state_msg.stream)
                    state_key = stream_state.stream_descriptor.name
                # looking how airbyte state finalizer works, only the last state is stored so we do the same
                # store the pydantic model
                state[state_key] = state_msg.dict()

            else:
                # Ignore unexpected or unhandled message types:
                # Type.LOG, Type.TRACE, Type.CONTROL, etc.
                pass

        # We are at the end of the stream. Process whatever else is queued.
        for stream_name, stream_batch in stream_batches.items():
            if len(stream_batch) > 0:
                yield {"stream_name": stream_name, "batch": stream_batch}


    @dlt.source(name=dlt_naming.normalize_identifier(ab_source.name))
    def read_airbyte():

        # create a reader resource that is deselected
        _read = dlt.resource(process_airbyte_messages(force_full_refresh), selected=False)

        def _stream_transformer(batch, stream_name=None):
            # Airbyte source is a monolith that sends data over UNIX pipe so we have
            # a single stream of data and here we route it from a single read dlt resource
            # to multiple transformers
            if batch["stream_name"] == stream_name:
                return batch["batch"]
            # allow the next one to pass
            return None

        # convert airbyte streams to dlt resources
        for stream in ab_source.configured_catalog.streams:
            # infer write disposition (not sure if fully correct)
            write_disposition: TWriteDisposition = "append"
            primary_key=stream.primary_key[0]
            if primary_key:
                write_disposition = "merge"

            if stream.destination_sync_mode == DestinationSyncMode.overwrite or stream.sync_mode == SyncMode.full_refresh or force_full_refresh:
                write_disposition = "replace"

            # NOTE: Airbyte has also cursor field which we could map into dlt incremental. but they handle all this stuff
            # in the source process so it is enough that we store state properly
            
            # print(write_disposition)
            # print(primary_key)
            # print(stream.destination_sync_mode)
            # print(stream.sync_mode)

            # streams are mapped into transformers, each transformer is attached to a read resource
            yield _read | dlt.transformer(
                name=stream.stream.name,
                write_disposition=write_disposition,
                primary_key=primary_key
            )(_stream_transformer)(stream_name=stream.stream.name)

    return read_airbyte()


source = ab.get_source(
    "source-pokeapi",
    config={"pokemon_name": "bulbasaur"},
    install_if_missing=True,
)
source.check()

pokeapi = convert_to_dlt_source(source, ["pokemon"])
# show resources
print(pokeapi)
# show initial schema
print(pokeapi.discover_schema().to_pretty_yaml())
# load to duckdb
pipeline = dlt.pipeline("pokemon", destination="duckdb", dataset_name="airbyte_test", full_refresh=True)
load_info = pipeline.run(pokeapi)
print(load_info)
# show inferred schema
print(pipeline.default_schema.to_pretty_yaml())
