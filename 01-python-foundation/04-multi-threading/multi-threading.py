## Multithreading
## Use this for I/O bound tasks

import threading
import time

def print_numbers():
    for i in range(5):
        time.sleep(2)
        print(f"Number:{i}")

def print_letters():
    for letter in "abcde":
        time.sleep(2)
        print(f"Letter: {letter}")


# creating threads
t1 = threading.Thread(target=print_numbers)
t2 = threading.Thread(target=print_letters)

t = time.time()
#Starting the threads
t1.start()
t2.start()

#waiting for the threads to complete and join the main thread
t1.join()
t2.join()
finished_time = time.time()-t
print(finished_time)


# when applied time.sleep(2) for both the functions, we see the time is more. As we are waiting
# for its execution to be completed. So we are going to use multithreading and see how its working.
# Number:0
# Number:1
# Number:2
# Number:3
# Number:4
# Letter: a
# Letter: b
# Letter: c
# Letter: d
# Letter: e
# 20.009626150131226

# when using multithreading
# Number:0Letter: a

# Letter: b
# Number:1
# Number:2Letter: c

# Letter: dNumber:3

# Letter: eNumber:4

# 10.074213027954102