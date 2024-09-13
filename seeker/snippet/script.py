#date: 2024-09-13T16:59:36Z
#url: https://api.github.com/gists/912a4c0e33d22e8ed5cba35a2f170296
#owner: https://api.github.com/users/kevinjqliu

import pyarrow.fs as fs

# List of URIs to test
file_uris = [
    "file:some/thing.csv",
    "file://some/thing.csv",
    "file:/some/thing.csv",
    "file:///some/thing.csv"
]

def test_file_uris(file_uris):
    for uri in file_uris:
        try:
            # Attempt to parse the URI
            filesystem, path = fs.LocalFileSystem.from_uri(uri)
            print(f"URI: {uri} -> Success: Filesystem: {filesystem}, Path: {path}")
        except Exception as e:
            print(f"URI: {uri} -> Error: {e}")

# Run the test
test_file_uris(file_uris)

# URI: file:some/thing.csv -> Error: File URI cannot be relative: 'file:some/thing.csv'
# URI: file://some/thing.csv -> Error: Unsupported hostname in non-Windows local URI: 'file://some/thing.csv'
# URI: file:/some/thing.csv -> Success: Filesystem: <pyarrow._fs.LocalFileSystem object at 0x104a10970>, Path: /some/thing.csv
# URI: file:///some/thing.csv -> Success: Filesystem: <pyarrow._fs.LocalFileSystem object at 0x1041e17b0>, Path: /some/thing.csv
