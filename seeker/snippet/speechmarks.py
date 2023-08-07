#date: 2023-08-07T16:52:25Z
#url: https://api.github.com/gists/49d58509ef4ae6dc76ec064838ec386c
#owner: https://api.github.com/users/schlarpc

import boto3
import enum
import json
import itertools
import streamp3
import dataclasses
from typing import Tuple, Dict, Optional


polly = boto3.client("polly")


class FileObjToIterator:
    def __init__(self, file_obj, size: Optional[int]=None):
        self._file_obj = file_obj
        self._size = size

    def __iter__(self):
        while chunk := self._file_obj(self._size):
            yield chunk

class IteratorToFileObj:
    def __init__(self, iterator):
        self._iterator = iter(iterator)
        self._buffer = bytearray()

    def read(self, size: Optional[int] = None) -> bytes:
        if size is None or size < 0:
            for chunk in self._iterator:
                self._buffer.extend(chunk)
            response = bytes(self._buffer)
            self._buffer = bytearray()
        else:
            if not self._buffer:
                try:
                    self._buffer.extend(next(self._iterator))
                except StopIteration:
                    pass
            response = self._buffer[:size]
            del self._buffer[:size]
        return bytes(response)


@dataclasses.dataclass
class PCMAudio:
    sample_rate: int
    channels: int
    sample_size: int
    data: bytes

class MP3Decoder:
    def __init__(self, mp3_byte_stream):
        self._mp3_byte_stream = mp3_byte_stream

    def __iter__(self):
        decoder = streamp3.MP3Decoder(IteratorToFileObj(self._mp3_byte_stream))
        for chunk in decoder:
            yield PCMAudio(
                sample_rate=decoder.sample_rate,
                channels=decoder.num_channels,
                sample_size=2,
                data=chunk,
            )


class SpeechMarkMP3Decoder:
    def __init__(self):
        ...

class MPEGVersion(enum.Enum):
    VERSION_1 = 0b11
    VERSION_2 = 0b10
    VERSION_2_5 = 0b00

class Layer(enum.Enum):
    LAYER_1 = 0b11
    LAYER_2 = 0b10
    LAYER_3 = 0b01


SAMPLES: Dict[Tuple[MPEGVersion, Layer], int] = {
    (version, layer): 1152 // {
        Layer.LAYER_1: 3,
        Layer.LAYER_2: 1,
        Layer.LAYER_3: (1 if version == MPEGVersion.VERSION_1 else 2),
    }[layer]
    for version, layer in itertools.product(MPEGVersion, Layer)
}

PADDING: dict[Layer, int] = {
    Layer.LAYER_1: 4,
    Layer.LAYER_2: 1,
    Layer.LAYER_3: 1,
}



# this is incomplete but covers at least everything Layer 3
BITRATE: dict[Tuple[MPEGVersion, Layer, int], int] = {
    **{
        (version, Layer.LAYER_3, idx): sum((
            (32 if version == MPEGVersion.VERSION_1 else 8),
            sum(8 << (_ // (4 if version == MPEGVersion.VERSION_1 else 7)) for _ in range(idx-1)),
        ))
        for version, idx in itertools.product(MPEGVersion, range(1, 15))
    },
    **{
        (MPEGVersion.VERSION_1, Layer.LAYER_1, idx): 32 * idx
        for idx in range(1, 15)
    },
}

FREQUENCY: dict[Tuple[MPEGVersion, int], int] = {
    (version, idx): [44100, 48000, 32000][idx] // {
        MPEGVersion.VERSION_1: 1,
        MPEGVersion.VERSION_2: 2,
        MPEGVersion.VERSION_2_5: 4,
    }[version]
    for version, idx in itertools.product(MPEGVersion, range(3))
}

ID3_HEADER_SIZE = 10
MP3_HEADER_SIZE = 4

if False:
    response = polly.synthesize_speech(
        Text='<speak><phoneme alphabet="ipa" ph="/jə kænt bi ˈduːɪn ðæt nəʊ mɔːr ɡɜːrl/"></phoneme></speak>',
        VoiceId="Kevin",
        Engine="neural",
        LanguageCode="en-US",
        OutputFormat="mp3",
        SpeechMarkTypes=["word"],
        TextType="ssml",
        SampleRate="16000",
    )
    print(response)
    with open("polly.mp3", "wb") as f:
        while chunk := response["AudioStream"].read(1024):
            f.write(chunk)

with open("polly.mp3", "rb") as f:
    data = bytearray(f.read())

def _parse_synchint(data):
    return sum([
        int(b) << (7 * index)
        for index, b in enumerate(data[::-1])
    ])

while data:
    if len(data) >= ID3_HEADER_SIZE and data[:3] == b"ID3":
        frame_size = ID3_HEADER_SIZE + _parse_synchint(data[6:10])
        if len(data) < frame_size:
            break
        id3_frame = data[ID3_HEADER_SIZE:frame_size]
        while id3_frame:
            tag_name = id3_frame[:4]
            tag_size = _parse_synchint(id3_frame[4:8])
            tag_data = id3_frame[ID3_HEADER_SIZE:tag_size+ID3_HEADER_SIZE]
            del id3_frame[:tag_size+ID3_HEADER_SIZE]
            if tag_name == b"TXXX":
                text_encoding = tag_data[0]
                assert text_encoding == 0x03
                strings = [s.decode("utf-8") for s in tag_data[1:].split(b"\x00")[:-1]]
                print(json.loads(strings[-1]))
    elif len(data) >= MP3_HEADER_SIZE and data[0] == 0xFF:
        mpeg_version = MPEGVersion((data[1] & 0b11000) >> 3)
        layer = Layer((data[1] & 0b110) >> 1)
        samples = SAMPLES[(mpeg_version, layer)]
        bitrate = BITRATE[(mpeg_version, layer, int(data[2]) >> 4)]
        frequency = FREQUENCY[(mpeg_version, (int(data[2]) & 0b1100) >> 2)]
        padding = PADDING[layer] if (int(data[2]) & 0b10) >> 1 else 0
        frame_size = (samples * bitrate // frequency // 8) + padding
        print(bitrate)
        if len(data) < frame_size:
            break
        mp3_frame = data[:frame_size]
        with open("pollystrip.mp3", "ab") as f: f.write(mp3_frame)
    del data[:frame_size]

with open("pollystrip.mp3", "rb") as f, open("pollystrip.pcm", "wb") as fout:
    for chunk in MP3Decoder(aaa()):
        fout.write(chunk.data)
