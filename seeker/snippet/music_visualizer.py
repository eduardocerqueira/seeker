#date: 2024-08-23T16:38:36Z
#url: https://api.github.com/gists/dca5714f15e5ca868a70d90d5c0dcab1
#owner: https://api.github.com/users/dnlzsy

# Author: Shubhang, 2023
# Description: This Python script demonstrates how to visualize audio using NumPy, Matplotlib, and MoviePy. 
# It reads an audio file in WAV format, converts the audio samples to a NumPy array, 
# and creates a video animation from a plot of the audio samples. 
# The resulting video file shows the amplitude of the audio samples over time.



import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage
from pydub import AudioSegment

def process_audio(file):
    audio = AudioSegment.from_wav(file)
    samples = np.array(audio.get_array_of_samples())
    return samples, audio.frame_rate

def visualize_music(samples, frame_rate, duration, output_file):
    fig, ax = plt.subplots()

    def make_frame(t):
        ax.clear()
        ax.set_title("Music Visualizer", fontsize=16)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")

        start = int(t * frame_rate)
        end = int((t + 1) * frame_rate)
        x = np.linspace(t, t + 1, end - start)
        y = samples[start:end]

        ax.plot(x, y, linewidth=0.5)
        ax.set_xlim(t, t + 1)
        ax.set_ylim(-2**15, 2**15 - 1)

        return mplfig_to_npimage(fig)

    animation = VideoClip(make_frame, duration=duration)
    animation.write_videofile(output_file, fps=30)

if __name__ == "__main__":
    input_file = "melody.wav"
    output_file = "music_visualizer.mp4"

    samples, frame_rate = process_audio(input_file)
    duration = len(samples) / frame_rate
    visualize_music(samples, frame_rate, duration, output_file)
