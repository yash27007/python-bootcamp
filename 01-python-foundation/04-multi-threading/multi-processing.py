## Multiprocessing Examples
## Process that run in parallel
## Use this for cpu bound tasks that are cpu heavy on cpu usage. 
## Parallel execution -- Multiple cores of the cpu

import multiprocessing
import time

def square_numbers():
    for i in range(5):
        time.sleep(1)
        print(f"Square: {i*i}")

def cube_numbers():
    for i in range(5):
        time.sleep(1.5)
        print(f"Cube: {i * i * i}")


if __name__ == "__main__":
    ## Create two processes

    p1 = multiprocessing.Process(target=square_numbers)
    p2 = multiprocessing.Process(target=cube_numbers)

    t=time.time()

    ## start the process

    p1.start()
    p2.start()


    # wait for the process to complete

    p1.join()
    p2.join()

    finished_time = time.time()-t
    print(finished_time) 


