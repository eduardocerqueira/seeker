#date: 2025-08-19T17:08:36Z
#url: https://api.github.com/gists/012738cb346e14a996281f35c5287676
#owner: https://api.github.com/users/EncodeTheCode

import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.signal import fftconvolve, get_window

def _load_wav(path):
    data, sr = sf.read(path, dtype='float32')
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, sr

def _softclip(x):
    return np.tanh(x)

def _generate_synthetic_ir(ir_wav, length_ms=300, decay=1.2, sr=44100):
    """
    Convert any WAV into a shaping IR:
    - Extract RMS envelope
    - Apply exponential decay
    - Smooth edges
    """
    length = int(sr * length_ms / 1000)
    ir_segment = ir_wav[:length]

    frame_size = 1024
    envelope = np.array([
        np.sqrt(np.mean(ir_segment[i:i+frame_size]**2))
        for i in range(0, len(ir_segment), frame_size)
    ])
    envelope = np.repeat(envelope, frame_size)[:length]

    decay_curve = np.exp(-np.linspace(0, decay, length))
    ir = envelope * decay_curve
    ir += np.random.randn(length) * 0.001  # small diffusion
    window = get_window('hann', length)
    ir *= window
    ir /= np.max(np.abs(ir)) + 1e-12
    return ir

def play_realtime(audio_path, ir_path=None, mix=0.5, volume=1.0,
                  ir_length_ms=300, ir_decay=1.2, blocksize=2048, grace_sec=2.0):
    """
    Play audio with real-time convolution reverb.
    Ensures full audio and reverb tail are heard by adding grace seconds.
    """
    audio, sr = _load_wav(audio_path)

    if ir_path:
        ir_wav, sr_ir = _load_wav(ir_path)
        if sr != sr_ir:
            raise ValueError("Sample rates must match")
        ir = _generate_synthetic_ir(ir_wav, length_ms=ir_length_ms, decay=ir_decay, sr=sr)
    else:
        ir = None

    # Add grace seconds to audio to let reverb tail decay
    grace_samples = int(grace_sec * sr)
    audio = np.concatenate([audio, np.zeros(grace_samples)])

    overlap = np.zeros(len(ir) if ir is not None else 0)
    idx = 0

    def callback(outdata, frames, time, status):
        nonlocal idx, audio, overlap
        chunk = audio[idx:idx+frames]
        idx += frames
        if len(chunk) < frames:
            chunk = np.pad(chunk, (0, frames-len(chunk)))

        if ir is not None:
            conv = fftconvolve(chunk, ir, mode='full')
            conv[:len(overlap)] += overlap
            if len(conv) > frames:
                overlap = conv[frames:]
            else:
                overlap = np.zeros_like(overlap)
            chunk_out = conv[:frames]
        else:
            chunk_out = chunk

        chunk_out = _softclip(chunk_out)
        chunk_out = np.clip(chunk_out * volume, -1.0, 1.0)
        outdata[:len(chunk_out), 0] = chunk_out
        if outdata.shape[1] > 1:
            outdata[:len(chunk_out), 1] = chunk_out

        if idx >= len(audio):
            raise sd.CallbackStop()

    with sd.OutputStream(channels=2, samplerate=sr, blocksize=blocksize, callback=callback):
        sd.sleep(int((len(audio)+len(overlap))/sr*1000)+500)

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    play_realtime(
        "sfx1.wav",
        "dome.wav",
        mix=0.3,
        volume=1.0,
        ir_length_ms=5000,
        ir_decay=1.5,
        grace_sec=3.0  # ensures full reverb tail is heard
    )
