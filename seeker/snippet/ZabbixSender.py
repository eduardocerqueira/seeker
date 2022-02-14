#date: 2022-02-14T16:59:20Z
#url: https://api.github.com/gists/6cf8586ba97b51200845045fbc373fe3
#owner: https://api.github.com/users/VietThan

import json

import socket
import struct
import time
import json

class ZabbixSender:
    '''
    Sender to Zabbix
    '''

    log = True

    def __init__(self, host='127.0.0.1', port=10051):
        '''
        init with default host and port of zabbix receiver
        data is empty list
        '''
        self.address = (host, port)
        self.data    = []

    def __log(self, log):
        if self.log: print(log)

    def __connect(self):
        '''
        Attempt connect with init address. 
        Helper for __request().
        Store socket in self.sock
        socket address/protocol family: AF_INET
        Socket type: SOCK_STREAM
        '''
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.sock.connect(self.address)
        except:
            raise Exception("Can't connect server.")

    def __close(self):
        '''
        close socket. Helper for __requests()
        '''
        self.sock.close()

    def __pack(self, request):
        '''
        pack zabbix request to send. Helper for __requests()
        convert from native Python string
        to string of bytes
        '''
        string = json.dumps(request)
        string_as_bytes= string.encode('utf-8')
        header = struct.pack('<4sBQ', b'ZBXD', 1, len(string_as_bytes))
        return header + string_as_bytes

    def __unpack(self, response):
        '''
        unpack zabbix response. Helper for __requests()
        from 
        '''
        header, version, length = struct.unpack('<4sBQ', response[:13])
        (data, ) = struct.unpack('<%ds'%length, response[13:13+length])
        return json.loads(data)

    def __request(self, request):
        '''
        Sends the request.
        Helper for send()
        '''
        
        # connect to self.address
        self.__connect()
        
        # try sending request
        try:
            self.sock.sendall(self.__pack(request))
        except Exception as e:
            raise
        
        # receive response
        response = b''
        while True:
            data = self.sock.recv(4096)
            if not data:
                break
            response += data
            
        # close connection
        self.__close()
        
        # returns unpacked response
        return self.__unpack(response)

    def __active_checks(self):
        '''
        Checks that the host is ready
        to receive data
        '''
        hosts = set()
        for d in self.data:
            hosts.add(d['host'])

        for h in hosts:
            request = {"request":"active checks", "host":h}
            self.__log("[active check] %s" % h)
            response = self.__request(request)
            if not response['response'] == 'success': 
                self.__log("[host not found] %s" % h)
                self.__log("response: %s" % response)

    def add(self, host, key, value, clock=None):
        if clock is None: clock = int(time.time())
        self.data.append({"host":host, "key":key, "value":value, "clock":clock})

    def send(self):
        '''
        send current list of data to specified address
        '''
        
        if not self.data:
            self.__log("Not found sender data, end without sending.")
            return False

        self.__active_checks()
        
        # craft and send request dict
        request  = {"request":"sender data", "data":self.data}
        response = self.__request(request)
        result   = True if response['response'] == 'success' else False
        
        # response handling
        if result:
            for d in self.data:
                self.__log("[send data] %s" % d)
            self.__log("[send result] %s" % response['info'])
        else:
            raise Exception("Failed send data.")

        return result

# The entrypoint function that is triggered with this lambda
# payload is passed along in event
def lambda_handler(event, context):
    sender = ZabbixSender()
    sender.add("gedowfather-example-01", "healthcheck", 1)
    sender.add("gedowfather-example-01", "gedow.item", 1111)
    sender.send()
    
    return {
        'statusCode': 200,
        'body': json.dumps(response)
    }