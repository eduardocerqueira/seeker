#date: 2023-02-27T17:02:11Z
#url: https://api.github.com/gists/9a5efcec0168634555a7c7da8617166d
#owner: https://api.github.com/users/c1cc10

#! /usr/bin/python
# -*- coding: utf-8 -*-

guitarix_pgm = "guitarix -p 7000"

import socket, json, os, time

class RpcNotification:

    def __init__(self, method, params):
        self.method = method
        self.params = params


class RpcResult:

    def __init__(self, id, result):
        self.id = id
        self.result = result


class RpcSocket:

    def __init__(self, address=("localhost",7000)):
        self.s = socket.socket()
        self.s.connect(address)
        self.buf = ""

    def send(self, method, id=None, params=[]):
        d = dict(jsonrpc="2.0", method=method, params=params)
        if id is not None:
            d["id"] ="1"
        toBeSent = json.dumps(d)+"\n"
        self.s.send(toBeSent.encode())

    def call(self, method, params=[]):
        self.send(method=method, id="1", params=params)

    def notify(self, method, params=[]):
        self.send(method=method, params=params)

    def receive(self):
        l = [self.buf]
        while True:
            p = l[-1]
            #print(p)
            if p == '':
                l.pop()
            #print(p.find('\n'))
            elif p.find(b"\n") > -1:
                ln, sep, tail = p.partition(b'\n')
                l[-1] = ln
                st = b"".join(l)
                self.buf = tail
                break;
            l.append(self.s.recv(10000))
        try:
            d = json.loads(st)
        except ValueError as e:
            print(e)
            print(st)
            return None
        if "params" in d:
            if not d["params"]:
                return None
            elif not  ".v" in (d["params"][0]):
                print(d["params"])
            return RpcNotification(d["method"], d["params"])
        elif "result" in d:
            return RpcResult(d["id"], d["result"])
        else:
            raise ValueError("rpc error: %s" % d)


class Guitarix():

    def open_socket(self):
        try:
            self.sock = RpcSocket()
        except socket.error as e:
            if e.errno != 111:
                raise
            return False
        return True

    def __init__(self):
        self.current_params = {}
        if not self.open_socket():
            
            os.system(guitarix_pgm+"&")
            for i in range(10):
                time.sleep(1)
                if self.open_socket():
                    break
            else:
                raise RuntimeError("Can't connect to Guitarix")
            self

def main():
    #start guitarix with rpc port at 7000
    gx = Guitarix()
    # open a socket at 7000
    sock = RpcSocket()
    # receive all available parameters from guitarix
    sock.call("parameterlist", [])
    parameterlist = []
    r = sock.receive().result
    for tp, d in zip(r[::2], r[1::2]):
        if tp == "Enum":
            d = d["IntParameter"]
        elif tp == "FloatEnum":
            d = d["FloatParameter"]
        d = d["Parameter"]
        n = d["id"]
        if "non_preset" in d and n not in ("system.current_preset", "system.current_bank"):
            continue
        parameterlist.append(d["id"])
    parameterlist.sort()
    # print out parameterlist
   # for i in parameterlist:
    #    print(i)
    
    # get current value of a parameter
    sock.call("get", ['wah.freq'])
    print(sock.receive().result)
    # set new value for a parameter
    sock.notify("set", ['wah.freq', 50])
    # and now listen to all parameter changes 
    sock.notify("listen",['all'])
    while sock:
        if sock.receive() == None:
            break

if __name__=="__main__":
    main()
