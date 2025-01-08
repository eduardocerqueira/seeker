#date: 2025-01-08T16:55:16Z
#url: https://api.github.com/gists/f140f5dd814c69a364b609a4d7b59e8f
#owner: https://api.github.com/users/Sg4Dylan

from __future__ import absolute_import, division, print_function, unicode_literals
import utils
import glob
import os
import numpy as np
import wandb
import random
import argparse
import librosa
import json
import torch
from scipy.io.wavfile import write
from env import AttrDict
from modules.waveumamba import waveumamba as Generator
from datetime import timedelta
from tqdm import tqdm
import time
from scipy.signal import butter, lfilter, sosfiltfilt
h = None
EPS = 1e-12

def prefix_load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint = torch.load(filepath, map_location=device)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    new_state_dict = {}
    prefix = "module."
    for key in state_dict["generator"].keys():
        if key.startswith(prefix):
            new_key = key[len(prefix):]  # Remove the prefix
            new_state_dict[new_key] = state_dict["generator"][key]

    print("Complete.")
    return new_state_dict

def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]

def wav_to_spectrogram(wav, hop_length, n_fft):
    f = np.abs(librosa.stft(wav, hop_length=hop_length, n_fft=n_fft, center = False))
    f = np.transpose(f, (1, 0))
    f = torch.tensor(f[None, None, ...])
    return f

def lsd(est ,target):
    assert est.shape == target.shape, "Spectrograms must have the same shape."
    est = est.squeeze(0).squeeze(0) ** 2
    target = target.squeeze(0).squeeze(0) ** 2
    # Compute the log of the magnitude spectrograms (adding a small epsilon to avoid log(0))
    epsilon = 1e-10
    log_spectrogram1 = torch.log10(target + epsilon)
    log_spectrogram2 = torch.log10(est + epsilon)
    squared_diff = (log_spectrogram1 - log_spectrogram2) ** 2
    squared_diff = torch.mean(squared_diff, dim = 1) ** 0.5
    lsd = torch.mean(squared_diff, dim = 0)
    return lsd

def load_audio(filepath, sr):
    y, sr = librosa.load(filepath, sr=sr, mono=False) # Load as multi-channel

    return y, sr

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    sos = butter(order, normal_cutoff, btype='high', output='sos', analog=False)
    return sos

def highpass_filter(data, cutoff, fs, order=5):
    sos = butter_highpass(cutoff, fs, order=order)
    y = sosfiltfilt(sos, data)
    return y

def infer(a, cfg, hf_freq, type=1):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.info["seed"])
        print("cuda initialized...")
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    random.seed(cfg.info["seed"])
    generator = Generator(cfg.mamba).to(device)
    print("model initialized...")
    state_dict_g = prefix_load_checkpoint(a.checkpoint_file, "cuda" if device.type == 'cuda' else "cpu")
    print("checkpoint successfully loaded...")
    generator.load_state_dict(state_dict_g, strict=True)

    max_inference_time = 30  # Maximum inference time in seconds
    overlap_len = 4800 # Define overlap length, can be adjusted. Should be small enough to not be noticeable to human ear
    hop_length = 48000*max_inference_time - overlap_len # Calculate hop length

    with torch.no_grad():
        y_low, sr_low = load_audio(a.wav_path, sr=48000)
        num_channels = y_low.shape[0]
        y_low_len = y_low.shape[1]

        output_audio = [[] for _ in range(num_channels)]
        
        # create overlap segments for each channel to help with alignment
        overlap_segments = [np.zeros(overlap_len) for _ in range(num_channels)]

        for channel in range(num_channels):
          start_idx = 0
          while start_idx < y_low_len:
              end_idx = min(start_idx + 48000 * max_inference_time, y_low_len)

              segment = y_low[channel, start_idx:end_idx]
              segment = torch.FloatTensor(utils.trim_or_pad(segment, 48000 * max_inference_time)).to(device)

              start_time = time.time()
              y_g_hat = generator(segment.unsqueeze(0).unsqueeze(0))
              end_time = time.time()

              audio_segment = y_g_hat.squeeze().cpu().numpy()

              # Apply high-pass filter
              filtered_segment = highpass_filter(audio_segment, hf_freq, 48000, order=5)

              # Mix filtered segment with original
              actual_len = y_low[channel, start_idx:end_idx].shape[0] # Get the actual length
              mixed_segment = np.zeros_like(audio_segment) # Create a zero array with the same shape as audio_segment
              mixed_segment[:actual_len] = filtered_segment[:actual_len] + y_low[channel, start_idx:end_idx]

              # Handle overlap for smoother transitions
              if start_idx > 0:
                  # Linear crossfade for the overlap region
                  fade_out = np.linspace(1, 0, overlap_len)
                  fade_in = np.linspace(0, 1, overlap_len)

                  mixed_segment[:overlap_len] = overlap_segments[channel] * fade_out + mixed_segment[:overlap_len] * fade_in
                  
              overlap_segments[channel] = mixed_segment[-overlap_len:]
              output_audio[channel].extend(mixed_segment[overlap_len:])
              
              inf_time = end_time - start_time
              print(f"Inference for channel {channel+1}, segment starting at {start_idx/48000}s took {inf_time:.5f} seconds")

              start_idx += hop_length

        audio = np.array(output_audio)

        output_file = os.path.join(a.output_dir, os.path.splitext(os.path.basename(a.wav_path))[0] + '_superresolved.wav')
        write(output_file, cfg.audio["sr"], audio.transpose())

        print(output_file)
        print("inference finished...")


def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--wav_path', default=None)
    parser.add_argument('--output_dir', default='/outputdir')
    parser.add_argument('--checkpoint_file', default = '/generator.pt')
    parser.add_argument('--cfgs_path', default='/ckpts.json')
    parser.add_argument('--hf_freq', type=float, default=19500, help='High-pass filter cutoff frequency')
    a = parser.parse_args()

    with open(a.cfgs_path, "r") as file:

        json_config = json.load(file)
        h = AttrDict(json_config)
        print("config initialized..")
    torch.manual_seed(h.info["seed"])
    
    infer(a, h, a.hf_freq)

if __name__ == '__main__':
    main()
    