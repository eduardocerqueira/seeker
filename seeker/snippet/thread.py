#date: 2023-02-16T16:38:31Z
#url: https://api.github.com/gists/9cc5f78ac6ccb7b69b9f5bcbaabb6b0e
#owner: https://api.github.com/users/Samg381

import threading
import time

Monitor = True

def thread_function():

    print("[Worker] worker created!")
    
    while (Monitor == True):
        print("[Worker] working...")
        time.sleep(1)
        
    print("[Worker] worker stopping!")


if __name__ == "__main__":

    print("[Main] Creating worker")
    
    x = threading.Thread(target=thread_function)
    x.start()
    
    time.sleep(8)
    
    print("[Main] Stopping worker")
    Monitor = False
    