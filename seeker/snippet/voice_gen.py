#date: 2024-05-13T16:47:05Z
#url: https://api.github.com/gists/6c0887032ccc6d3c0f0ca4bc4ed1e5e0
#owner: https://api.github.com/users/khalifaali

# Init TTS with the target model name
import torch
from TTS.api import TTS
device = "cuda" if torch.cuda.is_available() else "cpu"

OUTPUT_PATH='/out/path/'

tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False).to(device)

# Run TTS
tts.tts_to_file(text="Text goes here.",
                speaker_wav="/path/to/speaker/voice",
                file_path=OUTPUT_PATH,
                language="en")