import os
import threading
import time

def thread_function(batchID, threadID):
    ID = str(batchID)+str(threadID+2)
    os.system("python3 main.py -s " + ID + " -t 1000 -f 500 -b 50 -var 0.15 " + \
            " --output results/" + ID + "_var015" + "/")

# Runs many simulation in parallel
def batchRun(n):
    threads = list()
    for i in range(4):
        thread = threading.Thread(target=thread_function, args=(n, i))
        threads.append(thread)
        thread.start()

    for i in threads:
        i.join()

if __name__=="__main__":
    
    for i in range(125):
        batchRun(i+1)
        time.sleep(10)
