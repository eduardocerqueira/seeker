#date: 2025-05-19T16:53:13Z
#url: https://api.github.com/gists/fb70bd445e7b4b1b5360c4edfdf142f6
#owner: https://api.github.com/users/sennachereeb

import os

from twisted.internet import endpoints, reactor
from twisted.internet.endpoints import TCP4ServerEndpoint
from twisted.internet.protocol import Factory
from twisted.protocols.basic import LineReceiver


class EchoServer(LineReceiver):
    def connectionMade(self):
        print(f"EchoServer: client connected.")

    def connectionLost(self, reason):
        print(f"EchoServer: {reason.getErrorMessage()}")

    def lineReceived(self, line):
        print(f"EchoServer: received line -> {line}.")
        self.sendLine(line)
        print(f"EchoServer: sent line -> {line}.")


def main():
    factory: Factory = Factory.forProtocol(EchoServer)
    conn_string: str = (f"tcp:{int(os.environ.get('SERVER_PORT'))}:"
                        f"interface={os.environ.get('SERVER_HOST')}")

    endpoint: TCP4ServerEndpoint = (
        endpoints.serverFromString(reactor, conn_string))

    endpoint.listen(factory)

    print("LINE server started.")
    reactor.run()


if __name__ == "__main__":
    main()