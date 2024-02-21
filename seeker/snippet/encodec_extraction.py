#date: 2024-02-21T17:04:41Z
#url: https://api.github.com/gists/795d51cca0f8350ba67e9e64dcf34785
#owner: https://api.github.com/users/vatsalaggarwal

from audiocraft.data.audio import audio_read
from encodec import EncodecModel
import julius

audio_fname = "x.wav"

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MBD_SAMPLE_RATE = 24000
encodec = EncodecModel.encodec_model_24khz().to(DEVICE)
encodec.set_target_bandwidth(BANDWITDTH)

# read audio
wav, sr = audio_read(audio_fname)

# resample to mbd's expected sample rate
if sr != MBD_SAMPLE_RATE:
    wav = julius.resample_frac(wav, sr, MBD_SAMPLE_RATE)

# fix dimensionality, and convert to mono if needed
if wav.ndim == 2:
    wav = wav.unsqueeze(1)
    # if two channels, keep only one
    wav = wav[:1, :, :] if wav.shape[0] > 1 else wav

# extract tokens
wav = wav.to(DEVICE)
tokens = "**********"
tokens = "**********"= [8, T]][0].cpu().numpy()  # shape = [8, T]