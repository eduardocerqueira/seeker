#date: 2022-05-05T17:11:02Z
#url: https://api.github.com/gists/efc8e6d090343c3dc00380c1a5b4fd4a
#owner: https://api.github.com/users/plosso

import socket
import time

HOST = "palindromer-bd7e0fc867d57915.elb.us-east-1.amazonaws.com"  
PORT = 7777 
online = True
        
try:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(30)
        s.connect((HOST, PORT))
        while online:
            my_data = []
            final_list = []
            my_byte_data = b''
            data = s.recv(4096)
            print(f"Received {data!r}")


            #split the data to add as list element
            split = data.decode().split(' ')        
            for element in split:
                final_list.append(element.strip())
            
            #find palindromes
            for i in final_list:
                def isPalindrome(i):
                    return i == i[::-1]

                ans = isPalindrome(i)
                
                if ans:
                    my_data.append(i)
                    my_string  = ' '.join(my_data)
            
            my_string = my_string + "\n"
            my_byte_data = str.encode(my_string)
        

            #send my data back
            s.send(my_byte_data)    
            print(f"Sent {my_byte_data!r}")


            time.sleep(1)
    

except socket.error as e:
    print(e)
