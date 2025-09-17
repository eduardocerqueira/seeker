#date: 2025-09-17T16:59:21Z
#url: https://api.github.com/gists/81fb480985c2bf5206d2221eec542fb2
#owner: https://api.github.com/users/glowinthedark

#!/usr/bin/env python3
import sys
from pathlib import Path


# from: https://community.openai.com/t/transcribe-lines-are-way-too-long-for-both-subtitles-and-karaoke/289059/7

print('loading whisper...', end='')
import whisper
from whisper.utils import get_writer

print('Ok')

if __name__ == '__main__':
    
    # mac specific
    # import torch
    # print(f'MPS is available: {torch.backends.mps.is_available()}')

    model = whisper.load_model(name='large-v3')
    # will this ever be fixed??
    # model = whisper.load_model(name='large-v3', device="mps") # for macos, not working YET

    for audio in Path(".").glob(sys.argv[1]):

        srt_path = audio.with_suffix('.srt')
        if srt_path.exists():
            print(f'❗️ {srt_path.absolute()} already exists! skipping...')
            continue

        print(f'\n\n🚀 🚀 🚀 Processing:\n\t{audio.absolute()}...\n')
        result = model.transcribe(audio=str(audio),
                                temperature=0,
                                no_speech_threshold=0.8,
                                  hallucination_silence_threshold=1.0,
                                  beam_size=3,
                                  condition_on_previous_text=False,
                                  compression_ratio_threshold=2.4,
                                  task="transcribe",
                                  fp16=False,
#                                   word_timestamps=True,
#                                   temperature=0.1,
                                  verbose=True
                                  )

        # Set VTT Line and words width
        word_options = {
#             "highlight_words": True, ### << karaoke style subs
            "max_line_count": 3,
            "max_line_width": 42
        }

        # "txt": WriteTXT,
        # "vtt": WriteVTT,
        # "srt": WriteSRT,
        # "tsv": WriteTSV,
        # "json": WriteJSON,

        all_formats_writer = get_writer(output_format='all', output_dir='.')
        all_formats_writer(result, str(audio.absolute()), word_options)
     