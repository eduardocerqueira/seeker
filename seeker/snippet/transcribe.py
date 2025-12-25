#date: 2025-12-25T17:05:28Z
#url: https://api.github.com/gists/d660d4cfe5701cb5aecba9d441f0f05a
#owner: https://api.github.com/users/leonpahole

from faster_whisper import WhisperModel

model_size = "large-v3"

model = WhisperModel(model_size, device="cpu", compute_type="int8")

segments, info = model.transcribe("./HIB_1-18_The_Holiness_Code.mp3", beam_size=5)

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

stri = ''

for segment in segments:
    print(segment.text)
    stri += segment.text

with open("demofile.txt", "a") as f:
    f.write(stri)
