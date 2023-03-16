#date: 2023-03-16T16:45:58Z
#url: https://api.github.com/gists/b9a08f5db0ab098bb7e84a4cb8945668
#owner: https://api.github.com/users/khanh-nguyen-code

import asyncio

import cv2
import numpy as np
import websockets


async def receive_video():
    async with websockets.connect('ws://localhost:8765/') as socket:
        while True:
            buffer = await socket.recv()

            encoded_frame = np.frombuffer(buffer, np.uint8)
            decoded_frame = cv2.imdecode(encoded_frame, cv2.IMREAD_COLOR)

            cv2.imshow("output", decoded_frame)
            cv2.waitKey(1)


asyncio.run(receive_video())
