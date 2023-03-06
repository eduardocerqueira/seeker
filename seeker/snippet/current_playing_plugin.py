#date: 2023-03-06T16:50:21Z
#url: https://api.github.com/gists/802ab486374c69a183c85d5846100232
#owner: https://api.github.com/users/Sharrnah

# ============================================================
# Shows currently playing Song over OSC using Whispering Tiger
# See https://github.com/Sharrnah/whispering
# ============================================================

import Plugins

import VRC_OSCLib
import settings

# Install dependency using "pip install winsdk"

import asyncio
from winsdk.windows.media.control import GlobalSystemMediaTransportControlsSessionManager

PROMPT = {
    "command": ["playing", "listening", "listens", "song", "music", "track", "current"]
}


class CurrentPlayingPlugin(Plugins.Base):
    def __init__(self):
        print(self.__class__.__name__ + " loaded")
        pass

    async def get_current_song(self):
        osc_ip = settings.GetOption("osc_ip")
        osc_address = settings.GetOption("osc_address")
        osc_port = settings.GetOption("osc_port")

        if self.is_enabled():
            manager = await GlobalSystemMediaTransportControlsSessionManager.request_async()
            sessions = manager.get_sessions()
            for session in sessions:
                info = await session.try_get_media_properties_async()
                if info:
                    status = session.get_playback_info()
                    VRC_OSCLib.Chat(f"Currently {status.playback_status.name}: {info.title} by {info.artist}", True,
                                    False, osc_address,
                                    IP=osc_ip, PORT=osc_port,
                                    convert_ascii=False)
                    return info.title + " by " + info.artist
        return None

    def timer(self):
        if self.get_plugin_setting("timer", True):
            asyncio.run(self.get_current_song())
        pass

    def stt(self, text, result_obj):
        plugin_commands = self.get_plugin_setting("commands")
        if plugin_commands is None:
            plugin_commands = PROMPT['command']

        question = text.strip().lower()

        # return with current playing song if command word is found
        if any(ele in question for ele in plugin_commands):
            asyncio.run(self.get_current_song())

        return
