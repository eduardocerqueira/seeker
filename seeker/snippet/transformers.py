#date: 2026-03-06T17:13:54Z
#url: https://api.github.com/gists/38f85b0fdf73d902394668030265b122
#owner: https://api.github.com/users/simonespa

import sys
import numpy as np
from transformers import pipeline
from transformers.pipelines.audio_utils import ffmpeg_microphone_live
from curses import wrapper
import curses


pipe = pipeline("automatic-speech-recognition", model="openai/whisper-base", device=0)
sampling_rate = pipe.feature_extractor.sampling_rate

chunk_length_s = 5
stream_chunk_s = 0.1
mic = ffmpeg_microphone_live(
    sampling_rate=sampling_rate,
    chunk_length_s=chunk_length_s,
    stream_chunk_s=stream_chunk_s,  # , stride_length_s=(1, 0.1)
)
print("Start talking...")
stdscr = curses.initscr()
curses.noecho()
curses.cbreak()
text = ""
for item in pipe(mic):
    displayed = text + item["text"]
    if not item["partial"][0]:
        text += item["text"]

    stdscr.addstr(0, 0, displayed)
    stdscr.clrtoeol()
    stdscr.refresh()
