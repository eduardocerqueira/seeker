#date: 2023-04-10T17:01:31Z
#url: https://api.github.com/gists/c113f2fb68128c9ce505e922f8272e5b
#owner: https://api.github.com/users/lenforiee

# Apple Music RPC Server Bridge (Python)
# Author: @lenforiee <lenforiee@gmail.com>
# License: MIT

import time
import httpx
import logging
import uvicorn
import asyncio
from dataclasses import dataclass
from enum import IntEnum
from fastapi import FastAPI
from fastapi import Form
from pypresence import AioPresence

# Initialise logger.
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


CACHE = {
    "track_lookup": {},  # {track_name: track_id}
    "track_history": {},  # {track_id: Track}
    "track_playback_history": {},  # {track_id: [playback_time]}
    "user_state": None,  # UserState
}

QUEUE = []


@dataclass
class Track:
    arist: str
    title: str
    collecton: str
    track_id: int
    cover: str
    full_time: int


class PlaybackState(IntEnum):
    IDLE = 0
    PLAYING = 1
    PAUSED = 2

    def is_idle(self) -> bool:
        return self == PlaybackState.IDLE

    def is_playing(self) -> bool:
        return self == PlaybackState.PLAYING

    def is_paused(self) -> bool:
        return self == PlaybackState.PAUSED


@dataclass
class UserState:
    current_track: Track
    playback_state: PlaybackState
    rpc_showing: bool
    current_playback_time: int


async def get_song_metadata_from_apple_music(
    artist: str, title: str, album: str
) -> Track:
    url = f"https://itunes.apple.com/search"
    url_params = {
        "term": f"{artist} {title} {album}",
        "entity": "song",
        "media": "music",
        "limit": 1,
    }
    if album.find(artist) != -1:
        url_params.pop("limit", None)
    if album.find(title) != -1:
        url_params.pop("limit", None)

    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=url_params)
        data = response.json()

        if data["resultCount"] <= 1:
            data = data["results"][0]
        else:
            # Not always the first result is the correct one.
            for result in data["results"]:
                if (
                    result["trackName"] == title
                    and result["artistName"] == artist
                    and result["collectionName"] == album
                ):
                    data = result
                    break
            else:
                # If this is reached, it means something fucked up.
                assert False, "No result found."

        return Track(
            arist=data["artistName"],
            title=data["trackName"],
            collecton=data["collectionName"],
            track_id=data["trackId"],
            cover=data["artworkUrl100"],
            full_time=data["trackTimeMillis"],
        )


def parse_apple_time(t: str) -> int:
    if t.find(":") != -1:
        minute, second = t.split(":")
        return int(minute) * 60 + int(second)
    else:
        return int(t)


async def handle_close_update() -> str:
    # Check if there is different status on top of the queue.
    await asyncio.sleep(1)
    status = QUEUE[-1]
    if status != "close":
        return "The status is not closed."

    logger.info("Stopping the presence client...")
    await presence.clear()

    CACHE["track_history"] = {}
    CACHE["track_playback_history"] = {}
    CACHE["user_state"] = None

    return "Stopped the presence client"


async def handle_play_update(args: list[str]) -> str:
    if len(args) < 4:
        return "Invalid arguments"

    artist = args[0]
    title = args[1]
    album = args[2]

    current_playback_time = parse_apple_time(args[3])
    skipped = False
    skipped_timeline = False
    playing_again = False
    song_resumed = False

    track_id = CACHE["track_lookup"].get(f"{artist}+{title}+{album}")
    track: Track = CACHE["track_history"].get(track_id)
    if not track:
        track = await get_song_metadata_from_apple_music(artist, title, album)
        track_id = track.track_id
        CACHE["track_history"][track_id] = track
        CACHE["track_lookup"][f"{artist}+{title}+{album}"] = track_id

    user: UserState | None = CACHE["user_state"]
    if not user:
        user = UserState(
            current_track=track,
            playback_state=PlaybackState.IDLE,
            rpc_showing=False,
            current_playback_time=current_playback_time,
        )
        CACHE["user_state"] = user

    track_history: list[int] = CACHE["track_playback_history"].get(track_id)
    if not track_history:
        track_history = []
        track_history.append(current_playback_time)
    else:
        track_history.append(current_playback_time)

    if len(track_history) > 100:
        track_history.pop(0)

    CACHE["track_playback_history"][track_id] = track_history

    if (
        user.playback_state.is_playing()
        and track_id == user.current_track.track_id
        and len(track_history) > 3
        and track_history[-2] > current_playback_time
    ):
        # The song is played again.
        logger.info("The song is played again.")
        del CACHE["track_playback_history"][user.current_track.track_id]
        playing_again = True

    # We want to gather the first 3 seconds of playback time.
    if len(track_history) <= 3 and not playing_again:
        return "The song has not been played long enough."

    if track_history[-1] == track_history[-2] == track_history[-3]:
        # The song is paused.
        logger.info("The song has been paused.")

        if user.rpc_showing:
            await presence.clear()
            user.rpc_showing = False

        user.playback_state = PlaybackState.PAUSED
        user.current_playback_time = current_playback_time

        CACHE["user_state"] = user

        return "The song is paused."
    else:
        if user.playback_state.is_paused():
            logger.info("The song has been resumed.")
            song_resumed = True

    if track_id != user.current_track.track_id:
        logger.info("Another song has been played.")
        del CACHE["track_playback_history"][user.current_track.track_id]
        skipped = True

    if (
        track_id == user.current_track.track_id
        and track_history[-2] + 3 < current_playback_time
    ):
        # The song is skipped.
        logger.info("The timeline of song has been skipped.")
        skipped_timeline = True

    # The same song is playing.
    if not skipped:
        logger.info("The same song is playing.")
        user.current_playback_time = current_playback_time

        if (
            not skipped_timeline
            and not playing_again
            and user.playback_state.is_playing()
        ):
            CACHE["user_state"] = user
            return "The song is playing."

    if skipped:
        user.current_track = track
        user.current_playback_time = current_playback_time

    if (
        skipped
        or skipped_timeline
        or user.playback_state.is_idle()
        or playing_again
        or song_resumed
    ):
        logger.info("Updating the presence client...")

        if len(track.arist) < 2:
            track.arist = track.arist.rjust(2, " ")

        if len(track.title) < 2:
            track.title = track.title.rjust(2, " ")

        if len(track.collecton) < 2:
            track.collecton = track.collecton.rjust(2, " ")

        await presence.update(
            state=f"By {track.arist}",
            details=track.title,
            end=int(time.time())
            + (track.full_time // 1000 - current_playback_time + 2),
            large_text=track.collecton,
            large_image=track.cover,
        )
        user.playback_state = PlaybackState.PLAYING
        user.rpc_showing = True

    CACHE["user_state"] = user
    return "The song is playing."


def initialise_routes(app: FastAPI):
    @app.post("/update")
    async def update(message: str = Form(...)):
        logger.info(f"Received message: {message}")

        status, *args = message.split("|")

        if len(QUEUE) > 100:
            QUEUE.pop(0)

        match status:
            case "play":
                QUEUE.append("play")
                resp = await handle_play_update(args)
                return resp
            case "close":
                QUEUE.append("close")
                resp = await handle_close_update()
                return resp


def intialise_events(app: FastAPI):
    @app.on_event("startup")
    async def startup_event():
        logging.info("Initialising presence client...")
        await presence.connect()

    @app.on_event("shutdown")
    async def shutdown_event():
        logging.info("Shutting down the presence client...")
        presence.close()


def main() -> int:
    app = FastAPI()

    initialise_routes(app)
    intialise_events(app)

    uvicorn.run(app, host="0.0.0.0", port=13245)

    return 0


if __name__ == "__main__":
    presence = AioPresence("721415935424200735")
    raise SystemExit(main())
