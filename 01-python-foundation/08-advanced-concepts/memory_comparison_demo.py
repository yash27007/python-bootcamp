import sys
import time

N = 10_000_000

# Build a full list upfront
t0 = time.perf_counter()
full_list = [i * i for i in range(N)]
t1 = time.perf_counter()
list_build_time = t1 - t0
list_size = sys.getsizeof(full_list)

# Build a generator (lazy)
t0 = time.perf_counter()
gen = (i * i for i in range(N))
t1 = time.perf_counter()
gen_build_time = t1 - t0
gen_size = sys.getsizeof(gen)

print(f"N = {N:,}")
print(f"List build time:      {list_build_time:.4f}s")
print(f"Generator build time: {gen_build_time:.8f}s")
print(f"List sys.getsizeof:      {list_size:,} bytes ({list_size/1024/1024:.2f} MB)")
print(f"Generator sys.getsizeof: {gen_size:,} bytes")
print(f"Ratio: list is {list_size/gen_size:,.0f}x the size of the generator object")

# Time to sum via the full list vs consuming a fresh generator (fair comparison of full run)
t0 = time.perf_counter()
s1 = sum(full_list)
t1 = time.perf_counter()
sum_list_time = t1 - t0

t0 = time.perf_counter()
s2 = sum(i * i for i in range(N))
t1 = time.perf_counter()
sum_gen_time = t1 - t0

print(f"\nsum(full_list) result:            {s1}")
print(f"sum(generator expr) result:       {s2}")
print(f"Time to sum from prebuilt list:    {sum_list_time:.4f}s")
print(f"Time to sum via generator (no list build needed at all): {sum_gen_time:.4f}s")
