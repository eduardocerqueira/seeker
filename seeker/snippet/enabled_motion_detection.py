#date: 2025-05-07T17:03:19Z
#url: https://api.github.com/gists/1517adf98fba6db55cfc5f1b68a8cb5f
#owner: https://api.github.com/users/sideup66

import asyncio
from wyzeapy import Wyzeapy
from wyzeapy.services.camera_service import Camera, CameraService

async def async_main():
#Turns on motion detection
    client = await login()
    camera_service = await client.camera_service
    cameras = await camera_service.get_cameras()
    for camera in cameras:
        await camera_service.turn_on_motion_detection(camera)

#Define a login service
async def login() -> Wyzeapy:
    client = await Wyzeapy.create()
    #Syntax of client.login: "**********"
    await client.login("me@mail.com", "pass","key-id","api-key")
    return client


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(async_main())
  loop.run_until_complete(async_main())
