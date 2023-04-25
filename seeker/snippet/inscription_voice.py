#date: 2023-04-25T16:40:35Z
#url: https://api.github.com/gists/e7d94a22de0688fe9411c82aa67d2c55
#owner: https://api.github.com/users/SealtielFreak

import contextlib

import numpy as np
import pyaudio
from tensorflow import keras

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5
FILENAME_SAVE_MODEL = "inscription_voice_model.h5"


@contextlib.contextmanager
def open_pyaudio():
    _p = pyaudio.PyAudio()

    yield _p

    _p.terminate()


@contextlib.contextmanager
def open_stream(p, **kwargs):
    stream = p.open(**kwargs)

    yield stream

    stream.stop_stream()
    stream.close()


def find_working_microphone():
    with open_pyaudio() as p:
        for i in range(p.get_device_count()):
            device_info = p.get_device_info_by_index(i)

            if device_info['maxInputChannels'] > 0:
                try:
                    with open_stream(p, input_device_index=i, format=FORMAT, channels=CHANNELS, rate=RATE, input=True):
                        pass

                    return i
                except OSError:
                    continue

    return None


def recording_data_audio(record_seconds, output_device_index):
    frames = []

    with open_pyaudio() as p:
        with open_stream(p, format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK,
                         output_device_index=output_device_index) as stream:
            for i in range(0, int(RATE / CHUNK * record_seconds)):
                frames.append(stream.read(CHUNK))

    audio = np.frombuffer(b''.join(frames), dtype=np.int16)
    audio = audio.reshape((1, audio.shape[0], 1))

    return audio


def inscription_voice(record_seconds, output_device_index):
    print(f"Please speak for the user voice enrollment...")

    audio = recording_data_audio(record_seconds, output_device_index)

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(audio.shape[1], 1)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    print("Training voice model!")
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(audio, np.array([0]), epochs=10, verbose=0)
    model.save(FILENAME_SAVE_MODEL)

    return model


def check_voice(model, record_seconds, output_device_index):
    print("The voice belongs to the registered speaker...")

    audio = recording_data_audio(record_seconds, output_device_index)

    print("Checking voice!")
    prediction = model.predict(audio)

    if np.argmax(prediction) == 0:
        print("The voice belongs to the registered speaker")
    else:
        print("The voice does not belong to the registered speaker")


if __name__ == "__main__":
    working_microphone_index = find_working_microphone()

    print("Checking microphone...")
    if working_microphone_index is not None:
        print(f'The index of the working microphone is: {working_microphone_index}')
    else:
        print("No microphone was found that works")

    voice_model = inscription_voice(RECORD_SECONDS, working_microphone_index)

    check_voice(voice_model, RECORD_SECONDS, working_microphone_index)
