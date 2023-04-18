#date: 2023-04-18T16:46:01Z
#url: https://api.github.com/gists/7071f08d539bba6bd18e15ca40fc7c47
#owner: https://api.github.com/users/Sharrnah

# ============================================================
# Voicevox Text to Speech Plugin for Whispering Tiger
# V1.0.0
# See https://github.com/Sharrnah/whispering
# ============================================================
#
import base64
import io
import json
import sys
from importlib import util

import Plugins

import numpy as np
import pyaudio
import wave

from pathlib import Path
import os
import settings
import websocket
import downloader
import tarfile
import zipfile
import shutil


def load_module(package_dir):
    package_dir = os.path.abspath(package_dir)
    package_name = os.path.basename(package_dir)

    # Add the parent directory of the package to sys.path
    parent_dir = os.path.dirname(package_dir)
    sys.path.insert(0, parent_dir)

    # Load the package
    spec = util.find_spec(package_name)
    if spec is None:
        raise ImportError(f"Cannot find package '{package_name}'")

    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Remove the parent directory from sys.path
    sys.path.pop(0)

    return module


def extract_tar_gz(file_path, output_dir):
    with tarfile.open(file_path, "r:gz") as tar_file:
        tar_file.extractall(path=output_dir)
    # remove the zip file after extraction
    os.remove(file_path)


def extract_zip(file_path, output_dir):
    with zipfile.ZipFile(file_path, "r") as zip_file:
        zip_file.extractall(path=output_dir)
    # remove the zip file after extraction
    os.remove(file_path)


def move_files(source_dir, target_dir):
    for file_name in os.listdir(source_dir):
        source_path = os.path.join(source_dir, file_name)
        target_path = os.path.join(target_dir, file_name)

        # Check if it's a file
        if os.path.isfile(source_path):
            shutil.move(source_path, target_path)


voicevox_plugin_dir = Path(Path.cwd() / "Plugins" / "voicevox_plugin")
os.makedirs(voicevox_plugin_dir, exist_ok=True)

voicevox_core_python_repository = {
    "CPU": {
        "url": "https://github.com/VOICEVOX/voicevox_core/releases/download/0.14.3/voicevox_core-0.14.3+cpu-cp38-abi3-win_amd64.whl",
        "sha256": "02a3d7359cf4f6c86cc66f5fecf262a7c529ef27bc130063f05facba43bf4006"
    }
}
voicevox_core_dll_repository = {
    "CPU": {
        "url": "https://github.com/VOICEVOX/voicevox_core/releases/download/0.14.3/voicevox_core-windows-x64-cpu-0.14.3.zip",
        "sha256": "cf643566b08eb355e00b9b185d25f9f681944074f3ba1d9ae32bc04b98c3df50",
        "path": "voicevox_core-windows-x64-cpu-0.14.3"
    }
}
open_jtalk_dict_file = {
    "url": "https://jaist.dl.sourceforge.net/project/open-jtalk/Dictionary/open_jtalk_dic-1.11/open_jtalk_dic_utf_8-1.11.tar.gz",
    "sha256": "33e9cd251bc41aa2bd7ca36f57abbf61eae3543ca25ca892ae345e394cb10549",
    "path": "open_jtalk_dic_utf_8-1.11"
}


class VoicevoxTTSPlugin(Plugins.Base):
    core = None
    sample_rate = 16000
    acceleration_mode = "CPU"
    voicevox_core_module = None

    def init(self):
        # prepare all possible settings
        self.get_plugin_setting("speaker", 0)
        self.get_plugin_setting("acceleration_mode", "CPU")

        if self.is_enabled(False):
            print(self.__class__.__name__ + " is enabled")
            # disable default tts engine
            settings.SetOption("tts_enabled", False)

            self.acceleration_mode = self.get_plugin_setting("acceleration_mode", "CPU")

            if not Path(voicevox_plugin_dir / "voicevox_core" / "__init__.py").is_file():
                downloader.download_thread(voicevox_core_python_repository[self.acceleration_mode]["url"], str(voicevox_plugin_dir.resolve()), voicevox_core_python_repository[self.acceleration_mode]["sha256"])
                extract_zip(str(voicevox_plugin_dir / os.path.basename(voicevox_core_python_repository[self.acceleration_mode]["url"])), str(voicevox_plugin_dir.resolve()))

            if not Path(voicevox_plugin_dir / "voicevox_core" / "voicevox_core.lib").is_file():
                downloader.download_thread(voicevox_core_dll_repository[self.acceleration_mode]["url"], str(voicevox_plugin_dir.resolve()), voicevox_core_dll_repository[self.acceleration_mode]["sha256"])
                extract_zip(str(voicevox_plugin_dir / os.path.basename(voicevox_core_dll_repository[self.acceleration_mode]["url"])), str(voicevox_plugin_dir.resolve()))
                # move dll files to voicevox_core directory
                move_files(str(voicevox_plugin_dir / voicevox_core_dll_repository[self.acceleration_mode]["path"]), str(voicevox_plugin_dir / "voicevox_core"))

            open_jtalk_dict_path = Path(voicevox_plugin_dir / open_jtalk_dict_file["path"])
            if not Path(open_jtalk_dict_path / "sys.dic").is_file():
                downloader.download_thread(open_jtalk_dict_file["url"], str(voicevox_plugin_dir.resolve()), open_jtalk_dict_file["sha256"])
                extract_tar_gz(str(voicevox_plugin_dir / os.path.basename(open_jtalk_dict_file["url"])), str(voicevox_plugin_dir.resolve()))

            # load the voicevox_core module
            if self.voicevox_core_module is None:
                self.voicevox_core_module = load_module(str(Path(voicevox_plugin_dir / "voicevox_core").resolve()))

            if self.core is None:
                self.core = self.voicevox_core_module.VoicevoxCore(
                    acceleration_mode=self.acceleration_mode,
                    open_jtalk_dict_dir=str(open_jtalk_dict_path.resolve())
                )
        else:
            print(self.__class__.__name__ + " is disabled")
        pass

    def _generate_wav_buffer(self, audio):
        buff = io.BytesIO(audio)
        return buff

    def _play_audio(self, audio, device=None):
        buff = self._generate_wav_buffer(audio)

        # Set chunk size of 1024 samples per data frame
        chunk = 1024

        # Open the sound file
        wf = wave.open(buff, 'rb')

        # Create an interface to PortAudio
        p = pyaudio.PyAudio()

        # Open a .Stream object to write the WAV file to
        # 'output = True' indicates that the sound will be played rather than recorded
        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output_device_index=device,
                        output=True)

        # Read data in chunks
        data = wf.readframes(chunk)

        # Play the sound by writing the audio data to the stream
        while len(data) > 0:
            stream.write(data)
            data = wf.readframes(chunk)

        # Close and terminate the stream
        stream.close()
        wf.close()
        p.terminate()

    def predict(self, text, speaker):
        self.core.load_model(speaker)

        if len(text.strip()) == 0:
            return np.zeros(0).astype(np.int16)

        audio_query = self.core.audio_query(text, speaker)

        wav = self.core.synthesis(audio_query, speaker)

        return wav

    def play_tts(self, text):
        plugin_voice_name = self.get_plugin_setting("speaker", 0)

        return self.predict(text, plugin_voice_name)

    def timer(self):
        pass

    def stt(self, text, result_obj):
        if self.is_enabled(False) and settings.GetOption("tts_answer") and text.strip() != "":
            audio_device = settings.GetOption("device_out_index")
            wav = self.play_tts(text.strip())
            if wav is not None:
                self._play_audio(wav, audio_device)
        return

    def tts(self, text, device_index, websocket_connection=None, download=False):
        if self.is_enabled(False):
            if device_index is None:
                audio_device = settings.GetOption("device_out_index")
            else:
                audio_device = device_index

            wav = self.play_tts(text.strip())

            if wav is not None:
                if download and websocket_connection is not None:
                    wav_data = base64.b64encode(wav).decode('utf-8')
                    websocket.AnswerMessage(websocket_connection,
                                            json.dumps({"type": "tts_save", "wav_data": wav_data}))
                else:
                    self._play_audio(wav, audio_device)
        return

    def on_enable(self):
        self.init()
        pass

    def on_disable(self):
        pass
