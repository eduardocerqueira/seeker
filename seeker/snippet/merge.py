#date: 2022-12-28T17:07:48Z
#url: https://api.github.com/gists/7a0cc5f787629fdacc942cacb7786413
#owner: https://api.github.com/users/zironycho

from pydub import AudioSegment

audio1 = AudioSegment.from_wav('/path/to/input1.wav')
audio2 = AudioSegment.from_wav('/path/to/input2.wav')
# 0.3s silence
silence = AudioSegment.silent(duration=300)

output = audio1 + silence + audio2
output.export('/path/to/output.wav', format='wav')
