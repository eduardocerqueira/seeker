#date: 2023-09-25T17:03:22Z
#url: https://api.github.com/gists/4cdd9e479e46254371a4d207a674a0e5
#owner: https://api.github.com/users/james-see

from transformers import AutoProcessor, BarkModel
import scipy

processor = AutoProcessor.from_pretrained("suno/bark")
model = BarkModel.from_pretrained("suno/bark")

voice_preset = "v2/fr_speaker_2"

inputs = processor("[clears throat] Destroying is an act of creation, it makes room for new things to begin. Fire is the best method. Let the spark ignite your soul.", voice_preset=voice_preset)

audio_array = model.generate(**inputs)
audio_array = audio_array.cpu().numpy().squeeze()

sample_rate = model.generation_config.sample_rate
scipy.io.wavfile.write("destroy_bark_out_fr2.wav", rate=sample_rate, data=audio_array)

# run from terminal with aplay - afplay destroy_bark_out_fr2.wav -r 0.9