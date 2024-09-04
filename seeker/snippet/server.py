#date: 2024-09-04T16:38:11Z
#url: https://api.github.com/gists/79d97bee9e998322ae9610a082698857
#owner: https://api.github.com/users/jrgleason

import socket
import pyaudio
import wave
from openai import OpenAI

# Set up OpenAI client
client = OpenAI()

# Function to record audio
def record_audio(device_id, record_seconds=5, output_file="recorded_audio.wav"):
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 1024

    audio = pyaudio.PyAudio()

    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True, input_device_index=device_id,
                        frames_per_buffer=CHUNK)

    frames = []
    for _ in range(0, int(RATE / CHUNK * record_seconds)):
        data = stream.read(CHUNK)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    audio.terminate()

    with wave.open(output_file, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    return output_file

# Start TCP server using Wyoming Protocol
def start_wyoming_server(host='0.0.0.0', port=9000):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(1)
    print(f"Wyoming Protocol server listening on {host}:{port}")

    while True:
        client_socket, address = server_socket.accept()
        print(f"Connection from {address}")

        # Receive binary data (device_id and record_seconds)
        data = client_socket.recv(1024)
        device_id, record_seconds = data.split(b',')
        device_id = int(device_id)
        record_seconds = int(record_seconds)

        # Record audio
        audio_file = record_audio(device_id, record_seconds)

        # Transcribe the recorded audio
        with open(audio_file, "rb") as audio:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio
            )

        # Send the transcription result back to the client
        client_socket.sendall(transcription.text.encode('utf-8'))
        client_socket.close()

if __name__ == "__main__":
    start_wyoming_server()
