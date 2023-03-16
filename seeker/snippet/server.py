#date: 2023-03-16T16:45:58Z
#url: https://api.github.com/gists/b9a08f5db0ab098bb7e84a4cb8945668
#owner: https://api.github.com/users/khanh-nguyen-code

import asyncio

import cv2
import websockets

buffer: bytes = b""
interval: float = 0.03
encode_params: list[int] = [
    int(cv2.IMWRITE_JPEG_QUALITY), 90,
    int(cv2.IMWRITE_JPEG_PROGRESSIVE), 1,
    int(cv2.IMWRITE_JPEG_OPTIMIZE), 1,
]


async def send_video(socket, path):
    global buffer
    print(f"client connected: {socket}")
    try:
        while True:
            await socket.send(buffer)
            await asyncio.sleep(interval)
    except (websockets.exceptions.ConnectionClosedError, websockets.exceptions.ConnectionClosedOK) as e:
        print(f"client disconnected: {socket} {type(e)} {e}")


async def main():
    global buffer
    cap = cv2.VideoCapture(0)

    def read_and_encode() -> bytes:
        read_ok, frame = cap.read()
        if not read_ok:
            raise RuntimeError("read error")

        encode_ok, encoded_frame = cv2.imencode(".jpg", frame, encode_params)
        if not encode_ok:
            raise RuntimeError("encode error")

        return encoded_frame.tobytes()

    buffer = read_and_encode()

    video_server = await websockets.serve(send_video, "localhost", 8765)
    print(f"server started, listening on {video_server.sockets[0].getsockname()}")
    try:
        while True:
            buffer = read_and_encode()
            await asyncio.sleep(interval)
    except Exception as e:
        print(f"exception: {e}")
    finally:
        cap.release()


asyncio.run(main())
