#date: 2023-01-11T16:57:51Z
#url: https://api.github.com/gists/55abda7e48a7e05cab45abaf169b2461
#owner: https://api.github.com/users/AlexFernandes-MOVAI

""" plugin to handle udp """

import _thread
import re
import select
import socket
from time import sleep
from movai_service.movai_logger import log


class Handler:
    def __init__(self):

        self.name = "udp"

        self.commands = ["subscribe", "unsubscribe", "list"]

        # map port -> ip
        self._subs = {}
        self._stats = {}

        self._re_ip = re.compile(
            r'^(((2(([0-4][0-9])|(5[0-5])))|([0-1]?[0-9]{1,2}))\.){3}((2(([0-4][0-9])|(5[0-5])))|([0-1]?[0-9]{1,2}))$'
        )

        self._lock = _thread.allocate_lock()

    #
    # subscribe
    #
    # data received into port 'port'
    # is sento to ip:port
    #
    def subscribe(self, ip, port):
        need_unsubscribe = False
        try:
            port = int(port)
        except ValueError:
            log.error(f"Udp plugin: invalid {port}")
            return {"error": f"invalid port '{port}'"}

        if self._re_ip.match(ip) is None:
            # no match
            return {"error": f"invalid IP '{ip}'"}

        with self._lock:
            if port in self._subs:
                log.debug(f"Udp plugin: unsubscribe needed {port}")
                need_unsubscribe = True

        if need_unsubscribe:
            self.cancel_subscribe(port)

        with self._lock:
            self._subs[port] = ip
            self._stats[port] = {"src_ip": ip, "dest_ip": ip, "in": 0, "out": 0}

        _thread.start_new_thread(self._proxy, (port,))

        return {"status": "subscribed"}

    #
    # unsubscribe
    #
    # deprecated, removed the IP parameter use cancel_subscribe instead of this
    #
    def unsubscribe(self, ip, port):  # interface deprecated
        return self.cancel_subscribe(port)

    #
    # cancel_subscribe
    #
    # pretty much the reverse of subscribe
    #
    def cancel_subscribe(self, port):
        try:
            port = int(port)
        except ValueError:
            log.error(f"Udp plugin: invalid {port}")
            return {"error": f"invalid port '{port}'"}

        with self._lock:
            if port not in self._subs:
                return {"error": "unknown port"}

            del self._subs[port]
            del self._stats[port]
            # let time to the thread to die
            sleep(3)

        return {'status': 'unsubscribed'}

    def list(self):
        return {"status": self._stats}

    def _graceful_stop(self):
        with self._lock:
            self._subs.clear()
            self._stats.clear()

    #
    # internal functions
    #

    def _proxy(self, port):

        with self._lock:
            target = self._subs[port]

        # bind receiver with reuse option
        sock_rcv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, 0)
        sock_rcv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        addr_rcv = ('0.0.0.0', port)

        # bind sender
        sock_snd = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, 0)
        addr_snd = (target, port)
        sock_rcv.bind(addr_rcv)
        sock_rcv.setblocking(False)

        while True:
            # lock
            with self._lock:
                if port not in self._subs:
                    # no more subscribers
                    log.info(f"Udp plugin: breaking loop {port}")
                    break
            # release

            # wait for data
            rlist, wlist, xlist = select.select(
                sock_rcv,
                None,
                None,
                timeout=30.0)

            # receive from server
            if len(rlist):
                for sckt in rlist:
                    data, addr = rlist.recv(65535)

                # send to clients
                self._stats[port]['in'] += len(data)
                sent = sock_snd.sendto(data, addr_snd)
                self._stats[port]['out'] += sent

        # cleanup
        log.info(f"Udp plugin: closed {port}")
        sock_rcv.close()
        sock_snd.close()
