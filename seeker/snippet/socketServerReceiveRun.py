#date: 2021-09-16T16:50:29Z
#url: https://api.github.com/gists/ac567139208d9ece690143a289fef1d5
#owner: https://api.github.com/users/followthegik

#client
import socket
import subprocess


HOST = '127.0.0.1'  # The server's hostname or IP address//socket.gethostname()
PORT = 65432  
name = socket.gethostname() 
HOST = socket.gethostbyname(name)

s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
s.connect((HOST,PORT))
while True:
     command=s.recv(1024)
     if command == b'exit':
          s.close()
          break
     else:
        received = command.decode()
        print(received)
        cmd = []
        if received =='gaea-nodemap':
            cmd +=["C:/Users/graha/AppData/Local/QuadSpinner/Gaea/Gaea.Build.exe"]
            cmd +=["tor"]
            cmd +=["--nodemap"]
            cmd +=["--silent"]

        if received =='terragen':
            cmd +=["C:/Program Files/Planetside Software/Terragen 4/tgdcli.exe"]

        if len(cmd):
            print(cmd)
        #    proc = subprocess.Popen(" ".join(cmd), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE)
        #    output= proc.stdout.read()+proc.stderr.read()
        #    s.send(output)
        
        s.send(str.encode('heelo'))