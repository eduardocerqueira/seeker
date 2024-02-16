#date: 2024-02-16T16:53:03Z
#url: https://api.github.com/gists/5f5b865fccb220aae3ee8660b74101b5
#owner: https://api.github.com/users/0x48piraj

from socket import (
    socket, 
    AF_INET, 
    SOCK_STREAM,
    SO_REUSEADDR,
    SOL_SOCKET
)


class RequestParser(object):
    "Class for parsing request headers in a manageable fashion"

    def __init__(self, raw_request):
        self._raw_request = raw_request

        self._method, self._path, self._protocol, self._headers = self.parse()

    def parse(self):
        temp = [i.strip() for i in self._raw_request.splitlines()]

        # figuring out request method, path, HTTP protocol
        method, path, protocol = [i.strip() for i in temp[0].split()]

        # construct headers in case of a GET request
        headers = {}
        if 'GET' == method:
            for k, v in [i.split(':', 1) for i in temp[1:-1]]:
                headers[k.strip()] = v.strip()

        return method, path, protocol, headers

    def __repr__(self):
        return repr({'method': self._method, 'path': self._path, 'protocol': self._protocol, 'headers': self._headers})


HOST, PORT, CLRF = "127.0.0.1", 8080, "\r\n"
keystore = {}

with socket(AF_INET, SOCK_STREAM) as sock:
    sock.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
    sock.bind((HOST, PORT))
    sock.listen(1)
    while True:
        try:
            conn, addr = sock.accept()
            req = conn.recv(1024).decode()
            request = RequestParser(req)
            print(req)
            print("=" * 20)
            if '/set?' in request._path:
                k, v = request._path.split('?')[-1].split('=')
                keystore[k] = v
                conn.send(f'HTTP/1.1 200 OK{CLRF}'.encode())
                conn.send(f'Content-Type: text/html{CLRF*2}'.encode())
                conn.send(f'<h1>Stored key: {k}</h1>'.encode())
            elif '/get?' in request._path:
                k = request._path.split('?')[-1].split('=')[-1]
                print("Stored value:", keystore[k])
                conn.send(f'HTTP/1.1 200 OK{CLRF}'.encode())
                conn.send(f'Content-Type: text/html{CLRF*2}'.encode())
                conn.send(f'<h1>Stored Value: {keystore[k]}</h1>'.encode())
            else:
                conn.send(f'HTTP/1.1 200 OK{CLRF}'.encode())
                conn.send(f'Content-Type: text/html{CLRF*2}'.encode())
                conn.send('Hello world.'.encode())
            conn.close()
        except Exception as e:
            print(e)