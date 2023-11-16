#date: 2023-11-16T16:42:23Z
#url: https://api.github.com/gists/33b1ffa421886d6dcb973f9626f9f02d
#owner: https://api.github.com/users/githubway2us

# servermain.py
import socket
import threading

def handle_client(client_socket, client_address, connected_clients, sender_socket):
    print(f"Accepted connection from {client_address}")

    try:
        while True:
            data = client_socket.recv(1024)
            if not data:
                break
            print(f"Received from {client_address}: {data.decode('utf-8')}")

            # ส่งข้อมูลกลับไปที่ client
            client_socket.sendall(data)

            # ส่งข้อมูลไปยังเครื่องรับที่เชื่อมต่อ
            for receiver in connected_clients:
                if receiver != client_socket:
                    try:
                        receiver.sendall(data)
                    except ConnectionResetError:
                        print(f"Failed to send data to {receiver.getpeername()}")

            # ส่งข้อมูลไปยังเครื่องส่ง
            try:
                sender_socket.sendall(data)
            except ConnectionResetError:
                print(f"Failed to send data to sender")

    except ConnectionResetError:
        print(f"Connection with {client_address} forcibly closed by the remote host.")
    finally:
        connected_clients.remove(client_socket)
        client_socket.close()
        print(f"Connection from {client_address} closed")

def main():
    host = '0.0.0.0'  # ให้ server กลางรอรับ connection จากทุก IP address
    port = 8888  # กำหนดพอร์ต

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((host, port))
    server.listen(5)

    print(f"Listening on {host}:{port}...")

    connected_clients = []

    sender_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sender_socket.bind((host, 8889))  # กำหนดพอร์ตสำหรับเครื่องส่ง
    sender_socket.listen(1)

    sender, _ = sender_socket.accept()

    while True:
        client, addr = server.accept()
        connected_clients.append(client)

        client_handler = threading.Thread(target=handle_client, args=(client, addr, connected_clients, sender))
        client_handler.start()

if __name__ == "__main__":
    main()
