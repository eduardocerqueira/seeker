#date: 2024-05-02T16:59:31Z
#url: https://api.github.com/gists/4be889c448a26b7c4df669af4d496496
#owner: https://api.github.com/users/tyschacht

import sounddevice as sd
import numpy as np
import vosk
import queue

class VoiceRecorder:
    def __init__(self, model_path='model', device=None, activation_keyword='Hello Ada', end_keyword='thanks', stop_keyword='stop recording'):
        self.model = vosk.Model(model_path)
        self.device = device
        self.activation_keyword = activation_keyword.lower()
        self.end_keyword = end_keyword.lower()
        self.stop_keyword = stop_keyword.lower()
        self.interaction_transcript = ""
        self.recording = False
        self.q = queue.Queue()

    def callback(self, indata, frames, time, status):
        self.q.put(bytes(indata))

    def continuous_listen(self):
        with sd.RawInputStream(callback=self.callback, device=self.device, dtype='int16',
                               channels=1, samplerate=16000) as stream:
            rec = vosk.KaldiRecognizer(self.model, stream.samplerate)
            while True:
                data = self.q.get()
                if rec.AcceptWaveform(data):
                    result = rec.Result()
                    continue_listening = self.process_result(eval(result)['text'])
                    if not continue_listening:
                        print("Shutting down the listening process.")
                        break

    def process_result(self, transcript):
        print(f"Detected: {transcript}")
        if self.activation_keyword in transcript and not self.recording:
            self.start_interaction()
        elif self.end_keyword in transcript and self.recording:
            self.stop_interaction()
        elif self.stop_keyword in transcript:
            return False
        if self.recording:
            self.interaction_transcript += " " + transcript
        return True

    def start_interaction(self):
        print("Starting interaction ...")
        self.recording = True

    def stop_interaction(self):
        print("Stopping interaction ...")
        self.process_command(self.interaction_transcript)
        self.interaction_transcript = ""
        self.recording = False

    def process_command(self, transcript):
        # Process the recorded audio or perform actions based on the last command
        print(f"Processing command: {transcript}")


# Example usage:
if __name__ == "__main__":
# Ensure you have a Vosk model directory.
   recorder = VoiceRecorder('./audio_models/vosk-model-en-us-0.22-lgraph')
   recorder.continuous_listen()