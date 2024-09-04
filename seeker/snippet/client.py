#date: 2024-09-04T16:38:11Z
#url: https://api.github.com/gists/79d97bee9e998322ae9610a082698857
#owner: https://api.github.com/users/jrgleason

import socket

# Function to send request to the Wyoming server
def send_request_to_server(device_id, record_seconds):
    server_host = 'localhost'
    server_port = 9000

    # Establish connection to the server
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((server_host, server_port))

    # Send device_id and record_seconds as binary data
    request_data = f"{device_id},{record_seconds}".encode('utf-8')
    client_socket.sendall(request_data)

    # Receive transcription result
    transcription = client_socket.recv(4096).decode('utf-8')
    print("Transcription Result:", transcription)

    # Close the connection
    client_socket.close()

if __name__ == "__main__":
    # Ask the user for device ID and recording time
    device_id = int(input("Enter the device ID: "))
    record_seconds = int(input("Enter the number of seconds to record: "))

    # Send request to the Wyoming server
    send_request_to_server(device_id, record_seconds)