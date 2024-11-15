#date: 2024-11-15T16:52:57Z
#url: https://api.github.com/gists/be07e56aaed01ab6dc61e9e64bc8dfe6
#owner: https://api.github.com/users/jrkinley

import logging
import struct
from io import BytesIO
from confluent_kafka import Consumer
from datetime import datetime

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def read_int8(buffer):
    return struct.unpack(">b", buffer.read(1))[0]


def read_int16(buffer):
    return struct.unpack(">h", buffer.read(2))[0]


def read_int32(buffer):
    return struct.unpack(">i", buffer.read(4))[0]


def read_int64(buffer):
    return struct.unpack(">q", buffer.read(8))[0]


def read_string(buffer):
    len = read_int16(buffer)
    return buffer.read(len).decode("utf-8")


def read_optional(buffer):
    present = read_int16(buffer)
    if present < 0:
        return None
    return buffer.read(present).decode("utf-8")


def parse_message(key, value) -> tuple[dict, dict]:
    key_version, decoded_key = parse_key(key)
    decoded_value = {}
    if value:
        if key_version == 0 or key_version == 1:
            # offset commit value
            decoded_value = parse_offset_value(value)
        elif key_version == 2:
            # group metadata value
            decoded_value = parse_group_value(value)
    return decoded_key, decoded_value


def parse_key(key) -> tuple[int, dict]:
    rdr = BytesIO(key)
    key_version = read_int16(rdr)
    key_parsed = {"version": key_version}
    if key_version == 0 or key_version == 1:
        # offset commit key
        key_parsed["id"] = read_string(rdr)
        key_parsed["topic"] = read_string(rdr)
        key_parsed["partition"] = read_int32(rdr)
    elif key_version == 2:
        # group metadata key
        key_parsed["id"] = read_string(rdr)
    return key_version, key_parsed


def parse_offset_value(value) -> dict:
    rdr = BytesIO(value)
    value_parsed = {}
    version = read_int16(rdr)
    value_parsed["offset"] = read_int64(rdr)
    if version >= 3:
        value_parsed["leader_epoch"] = read_int32(rdr)
    value_parsed["metadata"] = read_string(rdr)
    value_parsed["timestamp"] = read_int64(rdr)
    if version == 1:
        value_parsed["expire_timestamp"] = read_int64(rdr)
    return value_parsed


def parse_group_value(value) -> dict:
    rdr = BytesIO(value)
    value_parsed = {}
    version = read_int16(rdr)
    value_parsed["protocol_type"] = read_string(rdr)
    value_parsed["generation_id"] = read_int32(rdr)
    value_parsed["protocol_name"] = read_optional(rdr)
    value_parsed["leader"] = read_optional(rdr)
    if version >= 2:
        value_parsed["state_timestamp"] = read_int64(rdr)
    value_parsed["members"] = read_int32(rdr)
    return value_parsed


conf = {
    "bootstrap.servers": "localhost:9092",
    "group.id": "test",
    "enable.auto.commit": "false",
    "auto.offset.reset": "smallest",
}
consumer = Consumer(conf)
consumer.subscribe(["__consumer_offsets"])
try:
    while True:
        msg = consumer.poll(timeout=1.0)
        if msg is None:
            continue
        if msg.error():
            print(msg.error())
            break
        key, val = parse_message(msg.key(), msg.value())

        msg = [f"v{key['version']}, group: {key['id']}"]
        if "topic" in key and "partition" in key:
            msg.append(f", topic: {key['topic']}, partition: {key['partition']}")
        if not val:
            msg.append(", [tombstone]")
        elif "offset" in val:
            msg.append(f", offset: {val['offset']}")
        elif "members" in val:
            msg.append(f", members: {val['members']}")
        logger.info("".join(msg))
except KeyboardInterrupt:
    print("Interrupted!")
finally:
    consumer.close()