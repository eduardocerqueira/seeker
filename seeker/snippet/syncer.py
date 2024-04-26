#date: 2024-04-26T16:58:39Z
#url: https://api.github.com/gists/1736a2688432c0ad7b3a7433d970464c
#owner: https://api.github.com/users/raymag

import os
import shutil
import subprocess
import ffmpeg
import librosa
import numpy as np
import time


def extract_peaks(y, sr):
    y, sr = librosa.load(audio_path)
    onset_env = librosa.onset.onset_strength(
        y=y, sr=sr, hop_length=256, aggregate=np.median
    )
    peaks = librosa.util.peak_pick(
        onset_env, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.5, wait=10
    )

    peak_times = librosa.frames_to_time(peaks, sr=sr)

    filtered_peaks = [0]
    for peak in peak_times:
        if peak - filtered_peaks[-1] >= 1.5:
            filtered_peaks.append(peak)

    return filtered_peaks


def load_song(audio_path):
    y, sr = librosa.load(audio_path)
    return y, sr


def cut_video(video_path, audio_path, peak_times):
    if os.path.isdir("temp"):
        shutil.rmtree("temp")
        os.mkdir("temp")
    else:
        os.mkdir("temp")

    if os.path.isfile("output.mp4"):
        os.remove("output.mp4")

    total_peaks = len(peak_times)
    i = 0
    chunk_i = 1
    MAX_CHUNKS = 200
    duration = 0
    input_paths = []
    while i + 2 < total_peaks - 1:
        if chunk_i > MAX_CHUNKS:
            break
        i += 2
        chunk_name = f"temp/chunk-{chunk_i}.mp4"

        subprocess.run(
            f"ffmpeg -ss {time.strftime('%H:%M:%S', time.gmtime(peak_times[i]))} -to {time.strftime('%H:%M:%S', time.gmtime(peak_times[i+1]))} -an -i {video_path} -c copy {chunk_name}",
            capture_output=True,
            check=True,
        )

        chunk_i += 1
        input_paths.append(chunk_name)
        duration += peak_times[i + 1] - peak_times[i]

    print("finished cutting chunks")

    command = f"ffmpeg -i {' -i '.join(input_paths)} -filter_complex \"concat=n={len(input_paths)}:v=1:a=0\" -y temp/output.mp4"
    subprocess.run(command, capture_output=True, check=True)

    subprocess.run(
        f"ffmpeg -ss 00:00:00 -to {time.strftime('%H:%M:%S', time.gmtime(duration))} -vn -i {audio_path} -c copy temp/audio.mp3",
        capture_output=True,
        check=True,
    )

    print(duration / 60)
    command = f"ffmpeg -i temp/output.mp4 -i temp/audio.mp3 -c:v copy -map 0:v -map 1:a output.mp4"
    subprocess.run(command, capture_output=True, check=True)

    shutil.rmtree("temp")

def main(audio_path, video_path):
    y, sr = load_song(audio_path)
    peak_times = extract_peaks(y, sr)

    cut_video(video_path, audio_path, peak_times)


if __name__ == "__main__":
    audio_path = "input_audio.mp3"
    video_path = "input_video.mp4"
    main(audio_path, video_path)
