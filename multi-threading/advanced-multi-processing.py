### Multi process with Process pool executor
from concurrent.futures import ProcessPoolExecutor
import time

def square_number(number):
    return f"Square: {number*number}"

numbers = [12,43,532,5342,23,1231,4,452343,56434564223]

if __name__ == "__main__":
    t = time.time()
    with ProcessPoolExecutor(max_workers=4) as executor:
        results = executor.map(square_number,numbers)
    
    for result in results:
        print(result)
    print("Time taken to complete: ",(time.time()-t), " seconds")
