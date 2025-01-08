#date: 2025-01-08T17:02:25Z
#url: https://api.github.com/gists/399ff93dd4a7d3acb71079df3b4608ea
#owner: https://api.github.com/users/brettschneider

"""Input/output from the from/to the user"""
import sys
from abc import ABC, abstractmethod
from io import BytesIO

import speech_recognition
from gtts import gTTS
from pyaudio import PyAudio
from pydub import AudioSegment


class CommandIO(ABC):
    @abstractmethod
    def get_input(self, prompt: str = None) -> str:
        """Retrieve command input from user"""

    @abstractmethod
    def report_output(self, output: str) -> None:
        """Respond to the user"""


class StdioCommandIO(CommandIO):
    """Console-based input/output"""

    def get_input(self, prompt="Command") -> str:
        try:
            return input(f"{prompt} > ")
        except (EOFError, KeyboardInterrupt):
            print("quit")
            return "quit"

    def report_output(self, output: str) -> None:
        print(output)


class AudioCommandIO(CommandIO):
    """Microphone/speaker-based input/output"""
    UNRECOGNIZABLE_AUDIO = "unrecognized audio"

    def __init__(self):
        self.recognizer = speech_recognition.Recognizer()

    def get_input(self, prompt='What would you like to do?') -> str:
        if prompt:
            self.report_output(prompt)
        with speech_recognition.Microphone() as mic:
            # self.recognizer.adjust_for_ambient_noise(mic)
            phrase = self.recognizer.listen(mic)
            while True:
                try:
                    text = self.recognizer.recognize_google(phrase)
                    print(f"I heard: {text}", file=sys.stderr)
                    return text
                except speech_recognition.UnknownValueError:
                    return self.UNRECOGNIZABLE_AUDIO
                except speech_recognition.RequestError as e:
                    print(f"Error: {e}")
                    return self.UNRECOGNIZABLE_AUDIO

    def report_output(self, output: str) -> None:
        mp3_fp = BytesIO()
        tts = gTTS(output, lang="en", slow=False)
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        audio_segment = AudioSegment.from_file(mp3_fp, format="mp3")
        pcm_data = audio_segment.raw_data
        pa = PyAudio()
        stream = pa.open(
            format=pa.get_format_from_width(audio_segment.sample_width),
            channels=audio_segment.channels,
            rate=audio_segment.frame_rate,
            output=True
        )
        stream.write(pcm_data)
        stream.stop_stream()
        stream.close()
