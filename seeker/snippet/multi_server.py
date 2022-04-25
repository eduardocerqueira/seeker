#date: 2022-04-25T17:16:03Z
#url: https://api.github.com/gists/b471b792abadf944d5268c7db4f5cba8
#owner: https://api.github.com/users/tiagocoutinho

import os
import socket
import logging


def cb(sock):
    reader = sock.makefile('rb')
    for line in reader:
        sock.sendall(line[::-1])


def accept(sock, log):
    while True:
        log.info("waiting for requests")
        conn, addr = sock.accept()
        log.info("client connected from %s", addr)
        try:
            cb(conn)
        except Exception as error:
            log.error("error: %r", error)
        finally:
            print("client disconnected from %r", addr)



def main():
    logging.basicConfig(level="INFO")
    sock = socket.create_server(("", 20_000), backlog=0)
    pid = os.fork()
    name = "parent" if pid else "child"
    log = logging.root.getChild(f"{name}({os.getpid()})")
    try:
        accept(sock, log)
    except KeyboardInterrupt:
        log.info("Ctrl-C pressed. Bailing out!")


if __name__ == "__main__":
    main()