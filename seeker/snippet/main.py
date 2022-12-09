#date: 2022-12-09T16:42:28Z
#url: https://api.github.com/gists/9b216a8a698b13d3bb56cc7b05d0205e
#owner: https://api.github.com/users/gustavojcd

#Importing library and thir function
from pydub import AudioSegment
from pydub.silence import split_on_silence
from sys import argv

#reading from audio mp3 file
filename = argv[1]
song = AudioSegment.from_mp3(filename)
save_path = './split_songs/'

# spliting audio files
audio_chunks = split_on_silence(song, min_silence_len=500, silence_thresh=-40 )

#loop is used to iterate over the output list
for i, chunk in enumerate(audio_chunks):
   output_file = save_path + "chunk{0}.mp3".format(i)
   print("Exporting file", output_file)
   chunk.export(output_file, format="mp3")

# chunk files saved as Output