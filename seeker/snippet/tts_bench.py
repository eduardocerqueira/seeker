#date: 2024-07-01T16:52:47Z
#url: https://api.github.com/gists/87b9dc0dd4c93d92e42de317f7190eb4
#owner: https://api.github.com/users/ryanhuang519

import statistics
import time
import wave

from abc import abstractmethod
from typing import Generator

import requests
from cartesia import Cartesia
from pyht import Client, Format, TTSOptions

fun_facts = [
    "Australia: The Great Barrier Reef is the largest living structure on Earth.",
    "Brazil: The Amazon rainforest produces 20 percent of the world's oxygen.",
    "Canada: Canada has the longest coastline of any country in the world.",
    "Denmark: Denmark is home to the oldest amusement park in the world, Bakken.",
    "Egypt: The ancient Egyptians invented the 365-day calendar.",
    "France: France is the most visited country in the world, attracting over 89 million tourists annually.",
    "Japan: Japan is home to more than 3,000 McDonald's restaurants, the most outside the US.",
    "Kenya: Kenya's Great Rift Valley was formed around 20 million years ago.",
    "Nepal: Mount Everest, the world's highest peak, is located in Nepal.",
    "Russia: Russia is the largest country in the world, covering more than 11 percent of Earth's land area.",
]


def benchmark_single(model: str, test_input: str):
    tts = TextToSpeechFactory().get(model)

    first_chunk_received = False
    start_time = time.time()
    first_chunk_time = None

    wav_file = wave.open(f"{model}_{test_input[:5]}.wav", "wb")
    wav_file.setnchannels(1)  # 1 channel
    wav_file.setsampwidth(2)  # 16-bit depth
    wav_file.setframerate(24000)

    for chunk in tts.full_to_stream(test_input):
        if not first_chunk_received:
            end_time = time.time()
            first_chunk_time = (end_time - start_time) * 1000
            first_chunk_received = True

        wav_file.writeframes(chunk)

    wav_file.close()
    return first_chunk_time


if __name__ == "__main__":
    for model in ["openai", "cartesia", "deepgram", "playht"]:
        times = []
        for sentence in fun_facts:
            times.append(benchmark_single(model, sentence))

        time_strs = [f"{t:.0f}ms" for t in times]
        print(f"[{model}] Times: {time_strs}")

        max_time = max(times)
        min_time = min(times)
        stddev_time = statistics.stdev(times)

        print(
            f"[{model}] Avg: {sum(times) / len(times):.0f}ms, Std: {stddev_time:.0f}ms, Max: {max_time:.0f}ms, Min: {min_time:.0f}ms"
        )



openai_key = "openai_key"
deepgram_key = "deepgram_key"
deepgram_key = "deepgram_key"

pyht_user = "pyht_user"
pyht_api_key = "pyht_api_key"


class TextToSpeech:
    def stream_to_stream(
        self, text_stream: Generator[str, None, None]
    ) -> Generator[bytes, None, None]:
        for text in text_stream:
            if text == "":
                continue

            for audio_chunk in self._stream_speech(text):
                yield audio_chunk
            yield text

    def full_to_stream(self, text: str) -> Generator[bytes, None, None]:
        for audio_chunk in self._stream_speech(text):
            yield audio_chunk

    def full_to_full(self, text: str) -> bytes:
        buffer = b""
        for chunk in self._stream_speech(text):
            buffer += chunk
        return buffer

    @abstractmethod
    def _stream_speech(self, text: str) -> Generator[bytes, None, None]:
        """
        asks the user_message to the LLM and calls the audio_callback with bytes that represent sound
        """
        pass


class TextToSpeechFactory:
    def get(self, model):
        if model == "openai":
            return OpenAITextToSpeech()
        if model == "deepgram":
            return DeepgramTextToSpeech()
        if model == "cartesia":
            return CartesiaTextToSpeech()
        if model == "playht":
            return PlayHTTextToSpeech()
        else:
            raise ValueError(f"Model {model} not supported")


class OpenAITextToSpeech(TextToSpeech):
    def _stream_speech(self, text: str) -> Generator[bytes, None, None]:
        url = "https://api.openai.com/v1/audio/speech"
        headers = {
            "Authorization": f"Bearer {openai_key}",  # Replace with your API key
        }

        data = {
            "model": "tts-1",
            "input": text,
            "voice": "alloy",
            "response_format": "pcm",
        }

        with requests.post(url, headers=headers, json=data, stream=True) as response:
            if response.status_code == 200:
                buffer = b""
                for chunk in response.iter_content(chunk_size=512):
                    buffer += chunk
                    while len(buffer) >= 512:
                        yield buffer[:512]
                        buffer = buffer[512:]
                if buffer:
                    yield buffer
            else:
                print(f"Error: {response.status_code} - {response.text}")


class DeepgramTextToSpeech(TextToSpeech):
    def _stream_speech(self, text: str) -> Generator[bytes, None, None]:
        DEEPGRAM_URL = "https://api.deepgram.com/v1/speak?model=aura-asteria-en&container=none&encoding=linear16"

        payload = {"text": text}

        headers = {
            "Authorization": "**********"
            "Content-Type": "application/json",
        }

        with requests.post(
            DEEPGRAM_URL, headers=headers, json=payload, stream=True
        ) as response:
            if response.status_code == 200:
                buffer = b""
                for chunk in response.iter_content(chunk_size=512):
                    buffer += chunk
                    while len(buffer) >= 512:
                        yield buffer[:512]
                        buffer = buffer[512:]
                if buffer:
                    yield buffer
            else:
                print(f"Error: {response.status_code} - {response.text}")


class CartesiaTextToSpeech(TextToSpeech):
    def __init__(self):
        self.cartesia_client = Cartesia(api_key=cartesia_key)
        self.ws = self.cartesia_client.tts.websocket()
        self.voice = self.cartesia_client.voices.get(
            id="a0e99841-438c-4a64-b679-ae501e7d6091"
        )

    def __del__(self):
        self.ws.close()

    def _stream_speech(self, text: str) -> Generator[bytes, None, None]:
        model_id = "sonic-english"

        output_format = {
            "container": "raw",
            "encoding": "pcm_s16le",
            "sample_rate": 24000,
        }

        # Generate and stream audio using the websocket
        for output in self.ws.send(
            model_id=model_id,
            transcript=text,
            voice_embedding=self.voice["embedding"],
            stream=True,
            output_format=output_format,
        ):
            yield output["audio"]


class PlayHTTextToSpeech(TextToSpeech):
    def __init__(self):
        self.play_client = Client(
            pyht_user,
            pyht_api_key,
        )

    def _stream_speech(self, text: str) -> Generator[bytes, None, None]:
        options = TTSOptions(
            voice="s3://voice-cloning-zero-shot/d9ff78ba-d016-47f6-b0ef-dd630f59414e/female-cs/manifest.json",
            format=Format.FORMAT_WAV,
        )

        for chunk in self.play_client.tts(
            text=text, voice_engine="PlayHT2.0-turbo", options=options
        ):
            yield chunk
yield chunk
