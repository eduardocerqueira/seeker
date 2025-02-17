#date: 2025-02-17T16:40:47Z
#url: https://api.github.com/gists/1c8d2fe434f4801d04ed291c3d44fe8f
#owner: https://api.github.com/users/REO2248

import ffmpeg
import sys
import kps9566

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python read_metadata.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    try:
        metadata = ffmpeg.probe(file_path, cmd='ffprobe')['format']['tags']
        print("Metadata for file: ", file_path)
        for key, value in metadata.items():
            value = value.encode('iso8859-1').decode('kps9566')
            print(f"  {key:14} :  {value}")
    except Exception as e:
        print("An error occurred: ", e)
        sys.exit(1)
