#date: 2023-12-01T16:38:06Z
#url: https://api.github.com/gists/a4ef33fe1325ed967bcf7bcbd18ad633
#owner: https://api.github.com/users/BeautyyuYanli

import socket
from typing import Optional


def http_once(
    host: str,
    port: int,
    buffer_size: int = 4096,
    timeout: Optional[float] = None,
    response: str = "OK",
) -> bytes:
    s = socket.create_server((host, port))
    s.settimeout(timeout)
    conn, _addr = s.accept()
    with conn:
        conn.settimeout(timeout)
        request = conn.recv(buffer_size)
        conn.sendall(response.encode("utf-8"))
    s.shutdown(socket.SHUT_RD)
    s.close()

    return request
