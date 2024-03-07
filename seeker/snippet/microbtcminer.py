#date: 2024-03-07T16:53:06Z
#url: https://api.github.com/gists/c397e03857368e7daab4931b15e01098
#owner: https://api.github.com/users/acaburaz

import socket
import json
import hashlib
import binascii
import urandom  # Alternative to the random module in Micropython
import network
import machine



# Concatenate "ESP" with MAC address
hostname = 'ESP-Miner'+'1'

# Print the hostname
print('Hostname:', hostname)

 

# Define the GPIO pin connected to the LED
LED_PIN = 2  # Assuming the LED is connected to GPIO pin 2

# Initialize the GPIO pin as an output
led = machine.Pin(LED_PIN, machine.Pin.OUT)

# Connect to your WiFi network
 "**********"d "**********"e "**********"f "**********"  "**********"c "**********"o "**********"n "**********"n "**********"e "**********"c "**********"t "**********"_ "**********"w "**********"i "**********"f "**********"i "**********"( "**********"s "**********"s "**********"i "**********"d "**********", "**********"  "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********") "**********": "**********"
    wlan = network.WLAN(network.STA_IF)
    wlan.config(dhcp_hostname=hostname)
    wlan.active(True)

    if not wlan.isconnected():
        print('Connecting to WiFi...')
        wlan.connect(ssid, password)
        while not wlan.isconnected():
            pass
    print('WiFi connected:', wlan.ifconfig())

# WiFi credentials
wifi_ssid = " "
wifi_password = "**********"

# Connect to WiFi
connect_wifi(wifi_ssid, wifi_password)


address = ''
nonce = '{:08x}'.format(urandom.getrandbits(32))
host = 'solo.ckpool.org'
port = 3333

def main():
    print("address:{} nonce:{}".format(address,nonce))
    print("host:{} port:{}".format(host,port))
   
    sock = socket.socket()
    addr = socket.getaddrinfo(host, port)[0][-1]
    sock.connect(addr)
   
    #server connection
    sock.sendall(b'{"id": 1, "method": "mining.subscribe", "params": []}\n')
    response = sock.recv(1024).decode()
    sub_details, extranonce1, extranonce2_size = json.loads(response)[u'result']
   
    #authorize workers
    sock.sendall(b'{"params": "**********": 2, "method": "mining.authorize"}\n')
   
    #we read until 'mining.notify' is reached
    response = b''
    while response.count(b'\n') < 4 and not(b'mining.notify' in response):
        response += sock.recv(1024)
   
    #get rid of empty lines
    responses = [json.loads(res) for res in response.decode().split('\n') if len(res.strip())>0 and 'mining.notify' in res]
    print(responses)
   
    job_id, prevhash, coinb1, coinb2, merkle_branch, version, nbits, ntime, clean_jobs = responses[0]['params']
   
    target = '{:0<64}'.format(nbits[2:] + '00' * (int(nbits[:2], 16) - 3))
    print('nbits:{} target:{}\n'.format(nbits,target))
   
    # extranonce2 = '00'*extranonce2_size
    extranonce2 = '{:0{}x}'.format(urandom.getrandbits(32), 2*extranonce2_size)
   
    coinbase = coinb1 + extranonce1 + extranonce2 + coinb2
    coinbase_hash_bin = hashlib.sha256(hashlib.sha256(binascii.unhexlify(coinbase)).digest()).digest()
   
    print('coinbase:\n{}\n\ncoinbase hash:{}\n'.format(coinbase,binascii.hexlify(coinbase_hash_bin)))
    merkle_root = coinbase_hash_bin
    for h in merkle_branch:
        merkle_root = hashlib.sha256(hashlib.sha256(merkle_root + binascii.unhexlify(h)).digest()).digest()
   
    merkle_root = binascii.hexlify(merkle_root).decode()
   
    #little endian
    merkle_root = ''.join([merkle_root[i]+merkle_root[i+1] for i in range(0,len(merkle_root),2)][::-1])
   
    print('merkle_root:{}\n'.format(merkle_root))
    led.on()

    def noncework():
        led.on()
        nonce = '{:08x}'.format(urandom.getrandbits(32))
        blockheader = version + prevhash + merkle_root + nbits + ntime + nonce + \
            '000000800000000000000000000000000000000000000000000000000000000000000000000000000000000080020000'
       
    #    print('blockheader:\n{}\n'.format(blockheader))
       
        hash = hashlib.sha256(hashlib.sha256(binascii.unhexlify(blockheader)).digest()).digest()
        hash = binascii.hexlify(hash).decode()
        # print('hash: {}'.format(hash))
        if hash[:5] == '00000': 
            print('hash: {}'.format(hash))
        if hash < target :
    #    if(hash[:10] == '0000000000'):
            print('success!!')
            print('hash: {}'.format(hash))
            payload = bytes('{"params": ["' + address + '", "' + job_id + '", "' + extranonce2 \
                + '", "' + ntime + '", "' + nonce + '"], "id": 1, "method": "mining.submit"}\n', 'utf-8')
            sock.sendall(payload)
            print(sock.recv(1024))
            #input("Press Enter to continue...")
    #    else:
    #        print('failed mine, hash is greater than target')
   
    while True:
        try:
            noncework()
        except:
            #led.off()
            machine.reset()
main()
