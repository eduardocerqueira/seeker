#date: 2024-02-22T17:01:04Z
#url: https://api.github.com/gists/d9d92fc222887a3126f386ef598cddcb
#owner: https://api.github.com/users/francescopapaleo

"""Hardware latency measurement using sounddevice API
Copyright (C) 2024 Francesco Papaleo

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import sounddevice as sd
import numpy as np


class HardwareLatencyMeasure:
    """
    A class used to measure the latency of an audio device.

    Parameters
    ----------
    device_index : int
        The index of the audio device to be tested.
    input_channel_index : int
        The index of the input channel to be used.
    output_channel_index : int
        The index of the output channel to be used.
    sample_rate : int, optional
        The sample rate to be used (default is 48000).
    duration : int, optional
        The duration of the signal in seconds (default is 5).
    pulse_width : float, optional
        The width of the pulse in seconds (default is 0.001).
    pulse_amplitude_dbfs : int, optional
        The amplitude of the pulse in dBFS (default is -1).
    """
    def __init__(self, device_index, input_channel_index, output_channel_index, sample_rate=48000, duration=5, pulse_width=0.001, amplitude_dbfs=-1):
        self.device_index = device_index
        self.input_channel_index = input_channel_index
        self.output_channel_index = output_channel_index
        self.sample_rate = sample_rate
        self.duration = duration
        self.start_time = duration / 2
        self.pulse_width = pulse_width
        self.amplitude_dbfs = amplitude_dbfs

    def dbfs_to_amplitude(self, dbfs):
        """Convert dBFS to a linear amplitude scale."""
        return 10 ** (dbfs / 20)
    
    def place_signal(self, signal, start_time, duration):
        """Place the signal at start_time within a duration of silence."""
        total_samples = int(self.sample_rate * duration)
        start_sample = int(self.sample_rate * start_time)
        silence_before = np.zeros(start_sample)
        silence_after = np.zeros(total_samples - len(signal) - len(silence_before))
        placed_signal = np.concatenate([silence_before, signal, silence_after])
        placed_signal = placed_signal.reshape(-1, 1)
        return placed_signal

    def generate_pulse_signal(self):
        """
        Generate a pulse signal for testing.

        Returns
        -------
        numpy.ndarray
            The generated pulse signal. Shape: (duration * sample_rate, 1)
        """
        pulse_amplitude = self.dbfs_to_amplitude(self.amplitude_dbfs)
        pulse = np.zeros(int(self.pulse_width * self.sample_rate))
        pulse[:] = pulse_amplitude
        placed_pulse = self.place_signal(pulse, self.start_time, self.duration)
        return placed_pulse
    
    def find_delay(self, original, recorded):
        """
        Find the delay between the original and recorded signals.

        Parameters
        ----------
        original : numpy.ndarray
            The original signal.
        recorded : numpy.ndarray
            The recorded signal.

        Returns
        -------
        float
            The delay time in seconds.
        int
            The delay index.
        """
        correlation = np.correlate(recorded, original, mode='full')
        delay_index = np.argmax(correlation) - len(original) + 1
        delay_time = delay_index / self.sample_rate
        return delay_time, delay_index
    
    def measure_latency(self):
        """
        Measure the latency of the audio device.

        Returns
        -------
        float
            The latency time in seconds.
        int
            The latency index.
        """
        playback_signal = self.generate_pulse_signal()
        recorded_signal = sd.playrec(playback_signal, samplerate=self.sample_rate, input_mapping=[self.input_channel_index], output_mapping=[self.output_channel_index], device=self.device_index, channels=1)
        sd.wait()  # Wait until recording is finished

        recorded_mono = recorded_signal[:, 0]
        latency_time, latency_index = self.find_delay(playback_signal[:, 0], recorded_mono)
        print(f"Estimated latency: {latency_time:.3f} seconds, Index: {latency_index}")
        return latency_time, latency_index


if __name__ == "__main__":
    print("Available audio devices:")
    print(sd.query_devices())
    device_index = int(input("Enter the index of the desired audio device: "))
    input_channel_index = int(input("Enter the input channel index: "))
    output_channel_index = int(input("Enter the output channel index: "))

    tester = HardwareLatencyMeasure(device_index, input_channel_index, output_channel_index)
    tester.measure_latency()
