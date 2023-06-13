#date: 2023-06-13T16:49:53Z
#url: https://api.github.com/gists/8bd2162cc2a3a2cb9ede8982416fa4d0
#owner: https://api.github.com/users/arturschaefer

import sys
import subprocess

def convert_webm_to_mp4(filename):
    try:
        subprocess.run(["ffmpeg", "-i", filename, "-c:v", "libx264", "-c:a", "aac", "-b:a", "192k", "-strict", "experimental", "-f", "mp4", filename[:-5] + ".mp4"], check=True)
    except subprocess.CalledProcessError as e:
        print("Error:", e)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert.py <filename.webm>")
        sys.exit(1)
    filename = sys.argv[1]
    if not filename.endswith(".webm"):
        print("Error: File must be a webm file.")
        sys.exit(1)
    convert_webm_to_mp4(filename)
    print("Conversion complete.")