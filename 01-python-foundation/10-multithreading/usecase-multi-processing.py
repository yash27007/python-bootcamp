"""
Real-world Example : Multiprocessing for CPU bound Task
Scenario : Factorial Calculation
Foctorial calculations, especially for larger numbers, 
involve significant computational work. 
Multiprocessing can be used to distribute the workload accross multiple
CPU cores, improving performance......
"""

import multiprocessing
import math
import sys
import time

# increasing the maximum number of digits for integer convesion

sys.set_int_max_str_digits(1000000)

# function to compute factorial of a given number

def compute_factorial(number):
    print(f"Computing factorial of {number}")
    result = math.factorial(number)
    return result

if __name__ == "__main__":
    numbers = [5000,2000,400,1343]
    start_time = time.time()

    ## create a pool f worker processes

    with multiprocessing.Pool() as pool:
        results = pool.map(compute_factorial,numbers)
    
    end_time = time.time()
    print(f"Results {results}")

    print(f"Total time: {(end_time - start_time)}")
