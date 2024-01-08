#date: 2024-01-08T16:58:42Z
#url: https://api.github.com/gists/0d39ec2b309c0ef7b055ca18be9584c7
#owner: https://api.github.com/users/edgarrmondragon

from typing_extensions import override

from singer_sdk import Stream, Tap


class ParentStream(Stream):
    name = "parent"
    schema = {
        "properties": {
            "W": {"type": "string"},
            "X": {"type": "string"},
        }
    }

    def get_response(self):
        return {
            "A": "",
            "B": "",
            "C": [
                {
                    "W": "",
                    "X": "",
                    "Y": [{"L": "", "M": "", "N": ""}],
                    "Z": [{"P": "", "Q": "", "R": ""}],
                }
            ],
        }

    @override
    def get_records(self, context):
        response = self.get_response()
        yield from response["C"]

    def get_child_context(self, record, context):
        return {
            "W": record["W"],
            "Y": record.pop("Y", []),
            "Z": record.pop("Z", []),
        }


class ChildStreamY(Stream):
    name = "child_y"
    schema = {
        "properties": {
            "W": {"type": "string"},
            "L": {"type": "string"},
            "M": {"type": "string"},
            "N": {"type": "string"},
        }
    }
    parent_stream_type = ParentStream
    state_partitioning_keys = ["W"]

    @override
    def get_records(self, context):
        yield from context["Y"]


class ChildStreamZ(Stream):
    name = "child_z"
    schema = {
        "properties": {
            "W": {"type": "string"},
            "P": {"type": "string"},
            "Q": {"type": "string"},
            "R": {"type": "string"},
        }
    }
    parent_stream_type = ParentStream
    state_partitioning_keys = ["W"]

    @override
    def get_records(self, context):
        yield from context["Z"]


class MyTap(Tap):
    name = "my-tap"

    def discover_streams(self):
        return [ParentStream(self), ChildStreamY(self), ChildStreamZ(self)]


if __name__ == "__main__":
    MyTap.cli()
