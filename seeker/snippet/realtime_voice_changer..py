#date: 2025-08-21T17:23:25Z
#url: https://api.github.com/gists/17f8045a088cafb7ed056a1aca19e847
#owner: https://api.github.com/users/grcubes

import pyaudio
import numpy as np
import librosa
import sys
import threading

class VoiceChanger:
    def __init__(self, pitch_shift=2, chunk_size=2048):
        self.CHUNK = chunk_size  # Reduce chunk size for lower latency
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 44100
        self.pitch_shift = pitch_shift
        self.p = pyaudio.PyAudio()
        self.running = True

        # Open input/output streams
        self.input_stream = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK
        )

        self.output_stream = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            output=True,
            frames_per_buffer=self.CHUNK
        )

    def process_audio(self, audio_chunk):
        # Convert bytes to numpy array
        audio_data = np.frombuffer(audio_chunk, dtype=np.float32)

        # Apply pitch shifting
        pitched = librosa.effects.pitch_shift(
            y=audio_data,
            sr=self.RATE,
            n_steps=self.pitch_shift,
            bins_per_octave=12
        )

        return pitched.astype(np.float32).tobytes()

    def audio_loop(self):
        while self.running:
            try:
                # Read audio chunk from microphone
                audio_chunk = self.input_stream.read(self.CHUNK, exception_on_overflow=False)

                # Process audio
                modified_audio = self.process_audio(audio_chunk)

                # Play modified audio
                self.output_stream.write(modified_audio)
            except Exception as e:
                print(f"Audio error: {e}")
                self.running = False

    def input_loop(self):
        print("Type 'up' or 'down' to adjust pitch, 'quit' to exit.")
        while self.running:
            command = input()
            if command == 'up':
                self.pitch_shift += 1
                print(f"Pitch shift increased to {self.pitch_shift}")
            elif command == 'down':
                self.pitch_shift -= 1
                print(f"Pitch shift decreased to {self.pitch_shift}")
            elif command == 'quit':
                self.running = False
            else:
                print("Unknown command.")

    def run(self):
        print("* Voice changer started. Type 'quit' to stop.")

        # Start audio and input threads
        audio_thread = threading.Thread(target=self.audio_loop)
        input_thread = threading.Thread(target=self.input_loop)
        audio_thread.start()
        input_thread.start()

        # Wait for threads to finish
        audio_thread.join()
        input_thread.join()

        self.cleanup()

    def cleanup(self):
        self.input_stream.stop_stream()
        self.input_stream.close()
        self.output_stream.stop_stream()
        self.output_stream.close()
        self.p.terminate()
        print("* Voice changer stopped.")

if __name__ == "__main__":
    # Get command line arguments for pitch shift and chunk size
    pitch_shift = 2
    chunk_size = 8192 # 2048, 4096, 8192...

    if len(sys.argv) > 1:
        try:
            pitch_shift = float(sys.argv[1])
        except ValueError:
            print("Invalid pitch shift value. Using default.")
    if len(sys.argv) > 2:
        try:
            chunk_size = int(sys.argv[2])
        except ValueError:
            print("Invalid chunk size value. Using default.")

    voice_changer = VoiceChanger(pitch_shift=pitch_shift, chunk_size=chunk_size)
    voice_changer.run()