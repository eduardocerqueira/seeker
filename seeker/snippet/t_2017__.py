#date: 2022-03-01T16:58:17Z
#url: https://api.github.com/gists/1a185a5ca729f4ee4be657d1d1a2f73f
#owner: https://api.github.com/users/belajarqywok

import os
import sys
import time
import json
import socket as sock

class auto:

    def __init__(self,VM):
        self.VM=VM
    def openVM(self):
        os.system(f"f:&cd services&cd {self.VM}&cd {self.VM}&{self.VM}.vbox")

class AboutThisTool:

    def theme(self):
        
        print(
            "=============================================\n"+
            "-------------- [ autoVM v1.0.0 ] ------------\n"+
            "=============== [ for windows ] =============\n"+
            "\n"+
            "[ -h ][ -- help ] ........... help\n"+
            "[ -lsVM ] ................ open directory VM\n"+
            "[ -openVM { VM name } ] ............ open single VM\n"+
            "[ -autorunVM { json file } ] ......... open all VM\n"+
            "[ -sshAccessService { json file } { service number } ] ......... open ssh service on VM"
        )

class Rules:

    def __init__(self):

        self.help=["-h","--help"]
        self.lsVM="-lsVM"
        self.openVM="-openVM"
        self.autorunVM="-autorunVM"
        self.sshAccessService="-sshAccessService"

    def run(self):

        try:
            if sys.argv[1] == self.lsVM:
                print("="*40)
                os.system("f:&dir services")
                print("="*40)

            elif (sys.argv[1] == self.help[0])|(sys.argv[1] == self.help[1]):
                AboutThisTool().theme()

            elif sys.argv[1] == self.openVM:
                auto(sys.argv[2]).openVM()

            elif sys.argv[1] == self.autorunVM:
                json_src=open(sys.argv[2],"r")
                result=json.load(json_src)

                for autorun in range(len(result["VMname"])):
                    loc=result["VMname"][f"service{str(autorun+1)}"]
                    os.system(f'{loc["src_OPEN"]}&{loc["name"]}')
                    print(f'=================================\n[ + ] status : RUNNING\n[ + ] VM name : {loc["name"][0:len(loc["name"])-5]}\n[ + ] VM location : {loc["src_OPEN"]}\n[ + ] running at [ {sock.gethostbyname(sock.gethostname())} ]')

            elif sys.argv[1] == self.sshAccessService:
                json_src=open(sys.argv[2],"r")
                result=json.load(json_src)
                serviceName=sys.argv[3]
                locationVMopen=result["VMname"][serviceName]

                os.system(f'{locationVMopen["src_OPEN"]}&{locationVMopen["name"]}')
                print("[ + ] PLEASE WAIT...................")
                time.sleep(60)

                if locationVMopen["SSH_access"]["host"]=="local":
                    os.system(f'ssh {locationVMopen["SSH_access"]["name"]}@{sock.gethostbyname(sock.gethostname())}')
                else:
                    os.system(f'ssh {locationVMopen["SSH_access"]["name"]}@{locationVMopen["SSH_access"]["host"]}')

            else:
                AboutThisTool().theme()

        except IndexError:
            AboutThisTool().theme()

        except FileNotFoundError:
            print(f"file {sys.argv[2]} not found......")

        except KeyError:
            print(f"{sys.argv[3]} unavailable!!, please chacking your service in {sys.argv[2]}")

        return True

# RUN
if __name__ == "__main__":

    os.system("cls")
    Rules().run()

# unittest
class testing(unittest.TestCase):

        def output_testing(self):
            self.assertEqual(Rules().run(),True)