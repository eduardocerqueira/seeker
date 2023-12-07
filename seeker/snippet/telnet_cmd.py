#date: 2023-12-07T16:54:37Z
#url: https://api.github.com/gists/eb0ced2fe5503ed9968921e5a0400d33
#owner: https://api.github.com/users/oschneidewind

#!/usr/bin/env python3
# While developing remote or daemon applications, I often feel the need for a
# simple Telnet-like console to quickly process a few commands. In the local
# console, there is the `cmd.Cmd` module from Python, which would actually be
# sufficient for my purposes. I was, therefore, wondering whether there is any
# way to parameterize the `Cmd` class so that it runs in the handle of
# socketserver and thus provides a simple interface to be addressed via netcat
# or Telnet.
#
# This Gist does exactly that; it connects a `cmd.Cmd` instance to a
# `StreamRequestHandler` to provide an easy way to interact with the current
# application. The Gist is more about showing basic feasibility than about secure
# or robust code. Therefore, no exceptions are handled, and secure connections
# are not used.
import cmd 
import io
import socketserver
import threading


class MenuCmd(cmd.Cmd):
    intro = "Welcome to the telnet style remote interface"
    prompt = "cmd> "

    def do_greeting(self, arg):
        self.stdout.write(f"hello {arg}\n")

    def do_close(self, arg):
        """
        close the connection
        """
        self.stdout.write("Bye Bye\n")
        return True

     
class MenuHandler(socketserver.StreamRequestHandler):

    def handle(self):
        rfile = io.TextIOWrapper(self.rfile, encoding="ascii", newline="\n")
        wfile = io.TextIOWrapper(self.wfile, encoding="ascii", newline="\n")
        menu = MenuCmd(stdin=rfile, stdout=wfile)
        menu.use_rawinput = False
        menu.cmdloop()


class ThreadedMenuServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    pass


def main():
    addr = ("localhost", 7777)
    server = ThreadedMenuServer(addr, MenuHandler)

    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()

    input(f"Server is listening to {addr}, press enter to exit")
    server.shutdown()


if __name__ == '__main__':
    main()      