import os
import threading
import time

def thread_function(batchID, threadID):
    adj = str(batchID)+str(threadID+2)
    os.system("python3 main.py -s 10 -t 1000 -f 500 -b 50 -var 0.15 -adj " + \
            "0." + adj + " --output results/verylowVaradj" + adj + "/")

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
    
    for i in range(2):
        batchRun(i)
