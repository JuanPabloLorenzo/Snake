import multiprocessing
import time
import numpy as np
import tensorflow as tf

def worker(buffer, i):
    for _ in range(3):
        identity = np.identity(200) 
        buffer.append(identity)
        print("put", i)

if __name__ == '__main__':
    buffer = multiprocessing.Manager().list()

    num_workers = 3
    processes = []

    for i in range(num_workers):
        process = multiprocessing.Process(target=worker, args=(buffer, i))
        processes.append(process)
        process.start()

    # Join all processes
    for process in processes:
        process.join()
