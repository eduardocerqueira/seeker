#date: 2023-07-31T17:07:51Z
#url: https://api.github.com/gists/5cc1bbdb2df9803eb1554a419dd24774
#owner: https://api.github.com/users/MiSaturo

import dns.message              #  --> pip install dnspython
import requests                 #  --> pip install requests

import dns.rdatatype
import base64
import socket
import time
import signal
import sys

def signal_handler(signal, frame):
    """ Catch Ctrl-C and Exit """
    sys.exit(0)

# Register SIGINT Handler
signal.signal(signal.SIGINT, signal_handler)


def dns_query(server_name):     

    
    quary_params = {
        # 'name': server_name,    # no need for this when using dns wire-format , cause 400 err on some server
        'type': 'A',
        'ct': 'application/dns-message',
        }
    

    try:
        query_message = dns.message.make_query(server_name,'A')
        query_wire = query_message.to_wire()
        query_base64 = base64.urlsafe_b64encode(query_wire).decode('utf-8')
        query_base64 = query_base64.replace('=','')    # remove base64 padding to append in url            

        query_url = DNS_url + query_base64
        ans = requests.get( query_url , params=quary_params , headers={'accept': 'application/dns-message'})
        
        # Parse the response as a DNS packet
        if ans.status_code == 200 and ans.headers.get('content-type') == 'application/dns-message':
            answer_msg = dns.message.from_wire(ans.content)

            resolved_ip = None
            for x in answer_msg.answer:
                if (x.rdtype == dns.rdatatype.A):
                    resolved_ip = x[0].address    # pick first ip in DNS answer
                    break
            
            return resolved_ip
        else:
            print(f'Error: {ans.status_code} {ans.reason}')
    except Exception as e:
        print(repr(e))
 
 
def tcp_ping(host, port=65533, timeout=1):
    s = socket.socket()
    s.settimeout(timeout)
    # s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    end = None
    try:
        start = time.time()
        s.connect((host, port))
        s.close()
        result = True
        end = time.time()
        ms = 1000*(end-start)
        return int(ms)
    except:
        return 0

   
    




# DNS_url = 'https://cloudflare-dns.com/dns-query?dns='      # not working in iran , ip blocked
# DNS_url = 'https://dns.google/dns-query?dns='              # not working in iran , ip blocked
# DNS_url = 'https://doh.opendns.com/dns-query?dns='           # not working in iran , ip blocked
# DNS_url = 'https://secure.avastdns.com/dns-query?dns='      # not working in iran , ip blocked
# DNS_url = 'https://doh.libredns.gr/dns-query?dns='          # not working in iran , ip blocked
DNS_url = 'https://dns.electrotm.org/dns-query?dns='
# DNS_url = 'https://dns.bitdefender.net/dns-query?dns='
# DNS_url = 'https://cluster-1.gac.edu/dns-query?dns='
 
host_s='sg1.ssh3.us sg2.ssh3.us sg3.ssh3.us sg4.ssh3.us us1.ssh3.us us2.ssh3.us us3.ssh3.us us4.ssh3.us us5.ssh3.us us6.ssh3.us us7.ssh3.us us8.ssh3.us hk1.ssh3.us au1.ssh3.us au2.ssh3.us jp1.ssh3.us jp2.ssh3.us in1.ssh3.us in2.ssh3.us gb1.ssh3.us gb2.ssh3.us de1.ssh3.us de2.ssh3.us fr1.ssh3.us fr2.ssh3.us fr3.ssh3.us pl1.ssh3.us ca1.ssh3.us ca2.ssh3.us sg1.tunnel01.xyz sg2.tunnel01.xyz sg3.tunnel01.xyz sg4.tunnel01.xyz sg5.tunnel01.xyz sg6.tunnel01.xyz sg7.tunnel01.xyz sg8.tunnel01.xyz us1.tunnel01.xyz us2.tunnel01.xyz us3.tunnel01.xyz us4.tunnel01.xyz us5.tunnel01.xyz us6.tunnel01.xyz us7.tunnel01.xyz us8.tunnel01.xyz kh1.tunnel01.xyz ua1.tunnel01.xyz ua2.tunnel01.xyz jp1.tunnel01.xyz jp2.tunnel01.xyz in1.tunnel01.xyz in2.tunnel01.xyz gp1.tunnel01.xyz gb2.tunnel01.xyz de1.tunnel01.xyz de2.tunnel01.xyz fr1.tunnel01.xyz fr2.tunnel01.xyz pl1.tunnel01.xyz ca1.tunnel01.xyz ca2.tunnel01.xyz sg1.ssh7.xyz sg2.ssh7.xyz sg3.ssh7.xyz sg4.ssh7.xyz us1.ssh7.xyz us2.ssh7.xyz us3.ssh7.xyz us4.ssh7.xyz us5.ssh7.xyz us6.ssh7.xyz us7.ssh7.xyz us8.ssh7.xyz hk1.ssh7.xyz au1.ssh7.xyz au2.ssh7.xyz jp1.ssh7.xyz jp2.ssh7.xyz in1.ssh7.xyz in2.ssh7.xyz gb1.ssh7.xyz gb2.ssh7.xyz de1.ssh7.xyz de2.ssh7.xyz fr1.ssh7.xyz fr2.ssh7.xyz fr3.ssh7.xyz pl1.ssh7.xyz ca1.ssh7.xyz ca2.ssh7.xyz sg1.ssh30.us sg2.ssh30.us sg3.ssh30.us sg4.ssh30.us us1.ssh30.us us2.ssh30.us us3.ssh30.us us4.ssh30.us us5.ssh30.us us6.ssh30.us us7.ssh30.us us8.ssh30.us hk1.ssh30.us au1.ssh30.us au2.ssh30.us jp1.ssh30.us jp2.ssh30.us in1.ssh30.us in2.ssh30.us gb1.ssh30.us gb2.ssh30.us de1.ssh30.us de2.ssh30.us fr1.ssh30.us fr2.ssh30.us fr3.ssh30.us pl1.ssh30.us ca1.ssh30.us ca2.ssh30.us sg1vip3.sshstores.vip sg2vip3.sshstores.vip sg3vip3.sshstores.vip sg4vip3.sshstores.vip sg5vip3.sshstores.vip sg6vip3.sshstores.vip sg7vip3.sshstores.vip sg8vip3.sshstores.vip us1vip3.sshstores.vip us2vip3.sshstores.vip us3vip3.sshstores.vip us4vip3.sshstores.vip us5vip3.sshstores.vip us6vip3.sshstores.vip us7vip3.sshstores.vip us8vip3.sshstores.vip hk1vip3.sshstores.vip au1vip3.sshstores.vip au2vip3.sshstores.vip jp1vip3.sshstores.vip jp2vip3.sshstores.vip in1vip3.sshstores.vip in2vip3.sshstores.vip gb1vip3.sshstores.vip gb2vip3.sshstores.vip de1vip3.sshstores.vip de2vip3.sshstores.vip fr1vip3.sshstores.vip fr2vip3.sshstores.vip pl1vip3.sshstores.vip ca1vip3.sshstores.vip ca2vip3.sshstores.vip sg1vip30.sshstores.vip sg2vip30.sshstores.vip sg3vip30.sshstores.vip sg4vip30.sshstores.vip sg5vip30.sshstores.vip sg6vip30.sshstores.vip sg7vip30.sshstores.vip sg8vip30.sshstores.vip us1vip30.sshstores.vip us2vip30.sshstores.vip us3vip30.sshstores.vip us4vip30.sshstores.vip us5vip30.sshstores.vip us6vip30.sshstores.vip us7vip30.sshstores.vip us8vip30.sshstores.vip hk1vip30.sshstores.vip au1vip30.sshstores.vip au2vip30.sshstores.vip jp1vip30.sshstores.vip jp2vip30.sshstores.vip in1vip30.sshstores.vip in2vip30.sshstores.vip gb1vip30.sshstores.vip gb2vip30.sshstores.vip de1vip30.sshstores.vip de2vip30.sshstores.vip fr1vip30.sshstores.vip fr2vip30.sshstores.vip pl1vip30.sshstores.vip ca1vip30.sshstores.vip ca2vip30.sshstores.vip' 


host_array=host_s.split(' ')
used = set()
unique_host_array = [x for x in host_array if x not in used and (used.add(x) or True)]

for host in unique_host_array:
    try:
        server_IP = dns_query(host)
        ping = tcp_ping(server_IP, timeout=1,port=80)
        if ping==0 :
            continue
        print(f"{host} -> {ping}ms")
    except:
        pass
