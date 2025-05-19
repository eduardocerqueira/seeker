#date: 2025-05-19T17:04:54Z
#url: https://api.github.com/gists/03d40900004747a41e145fe9fb74c481
#owner: https://api.github.com/users/sennachereeb

import os

from twisted.internet import reactor
from twisted.internet.defer import Deferred, inlineCallbacks
from twisted.internet.protocol import ClientFactory
from twisted.internet.task import react
from twisted.protocols.basic import LineReceiver
from txtorcon import endpoints, TorClientEndpoint


class EchoClient(LineReceiver):
    line = b"Hello!"

    def connectionMade(self):
        addr = self.transport.getPeer()
        print(f"EchoClient: connection made to {addr.host}:{addr.port}.")
        self.sendLine(self.line)
        print(f"EchoClient: sent line -> {self.line}.")

    def lineReceived(self, line):
        print(f"EchoClient: received line -> {line}")


class EchoClientFactory(ClientFactory):
    def __init__(self):
        self.protocol = EchoClient
        self.done = Deferred()

    def clientConnectionFailed(self, connector, reason):
        print("connection failed:", reason.getErrorMessage())
        self.done.errback(reason)

    def clientConnectionLost(self, connector, reason):
        print("connection lost:", reason.getErrorMessage())
        self.done.callback(None)


def main():
    conn_string = (f"tor:host={os.environ.get('HIDDEN_HOST')}:"
                   f"port={int(os.environ.get('SERVER_PORT'))}:"
                   f"socksHostname={os.environ.get('SOCKS5_HOST')}:"
                   f"socksPort={int(os.environ.get('SOCKS5_PORT'))}")

    onion_endpoint: TorClientEndpoint = (
        endpoints.clientFromString(reactor, conn_string))

    client_factory = EchoClientFactory()
    onion_endpoint.connect(client_factory)
    
    reactor.run()


if __name__ == "__main__":
    main()
