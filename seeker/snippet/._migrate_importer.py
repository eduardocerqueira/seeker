#date: 2023-02-15T17:08:22Z
#url: https://api.github.com/gists/8f3d7517450b399a045b950c1799a972
#owner: https://api.github.com/users/NiltonDuarte

from ruamel.yaml import YAML
from pathlib import Path
import os

curr_folder = Path(__file__).parent
yaml = YAML()


class QueryNotFoundError(Exception):
    pass


def replace_stream(content):
    """
    export_local:
      stream: Prod-<stream>
    """
    stream = content['export_local']['stream']
    content['export_local']['stream'] = stream.replace("Prod-", "").replace("Stage-", "")


def replace_connection(content):
    """
    connections:
    - type: other
      alias: rivers_kinesis
      export_to: rivers
    """
    for connection in content["connections"]:
        connection["alias"] = connection["alias"].replace("rivers_kinesis", "rivers_kafka")


def add_metadata_field(content):
    """
    steps:
      - name: loading
        configuration:
            transformer:
                query: >-
                    SELECT
                    date(metadata.event_metadata.time) as p_event_date,
                    metadata.event_metadata.time as event_time,
                    base64(metadata.event_metadata.nonce) as nonce,
                    `data__.*`
                    FROM <SRC>
    """
    for step in content["steps"]:
        if step["selector"]["source"] == "rivers":
            transformer = step["configuration"]["transformer"]
            transformer["query"] = transformer["query"].replace("`data__.*`", "metadata, `data__.*`")
            return
    else:
        raise QueryNotFoundError()


def main():
    for file_name in os.listdir(curr_folder):
        if not file_name.endswith('.initial.yaml'):
            continue
        print(f"Migrating {file_name}")
        with open(curr_folder / file_name, 'r+') as stream:
            content = yaml.load(stream)
            try:
                replace_stream(content)
                replace_connection(content)
                add_metadata_field(content)
            except KeyError:
                print(f"Skipping {file_name}")
                continue
            stream.truncate(0)
            stream.seek(0)
            yaml.dump(content, stream)
        # return


main()
