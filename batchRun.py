import os
import threading
import time

def thread_function():
    os.system("python main.py -t 1000 -f 500 -b 50 -var 0.1")

# Runs many simulation in parallel
def batchRun():
    threads = list()
    for i in range(4):
        thread = threading.Thread(target=thread_function)
        threads.append(thread)
        thread.start()

    for i in threads:
        i.join()

if __name__=="__main__":
    
    for i in range(10):
        batchRun()
