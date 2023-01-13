#date: 2023-01-13T16:56:14Z
#url: https://api.github.com/gists/5e44cf2d4affc99bc0e775cb636443be
#owner: https://api.github.com/users/schtschenok

"""
    This script adds the necessary data to a mono WAV file for Wwise to recognize it as a single-channel LFE file.

    Requires: wave-chunk-parser library (https://pypi.org/project/wave-chunk-parser/)
    Usage: python wwise_lfe.py /path/to/file.wav
    Results: a new WAV file in the same directory as the input one, with .LFE.wav extension (such as file.LFE.wav)

    Copyright 2023 Tyoma Makeev

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        https://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""

import sys
from pathlib import Path

from wave_chunk_parser.chunks import FormatChunk, RiffChunk, WaveFormat

lfe_extension = b'\x10\x00\x08\x00\x00\x00\x01\x00\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x008\x9bq'


def main():
    if len(sys.argv) != 2:
        print("Please specify a path to a single mono WAV file as an argument")
        sys.exit(1)

    input_file = Path(sys.argv[1])

    with open(input_file, "rb") as file:
        riff_chunk = RiffChunk.from_file(file)
        format_chunk_index = 0
        for index, chunk in enumerate(riff_chunk.sub_chunks):
            if isinstance(chunk, FormatChunk):
                format_chunk_index = index
                break
        format_chunk = riff_chunk.sub_chunks[format_chunk_index]
        if format_chunk.channels != 1:
            raise ValueError("You should only use mono WAV files.")
        new_format_chunk = FormatChunk(WaveFormat.EXTENDED,
                                       lfe_extension,
                                       format_chunk.channels,
                                       format_chunk.sample_rate,
                                       format_chunk.bits_per_sample,
                                       format_chunk.byte_rate,
                                       format_chunk.block_align)
        riff_chunk.sub_chunks[format_chunk_index] = new_format_chunk
        output_file = input_file.parent / f"{input_file.stem}.LFE{input_file.suffix}"
        with open(output_file, "wb") as new_file:
            new_file.write(RiffChunk(riff_chunk.sub_chunks).to_bytes())
            print(f"Written to {output_file.relative_to(input_file.parent)}")


if __name__ == "__main__":
    main()
