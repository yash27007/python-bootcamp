import time
import threading

NUM_THREADS = 4
INCREMENTS_PER_THREAD = 2000
EXPECTED = NUM_THREADS * INCREMENTS_PER_THREAD


def run_unsafe():
    """Two+ threads incrementing a shared counter with NO lock."""
    counter = 0

    def worker():
        nonlocal counter
        for i in range(INCREMENTS_PER_THREAD):
            temp = counter                # READ counter
            if i % 500 == 0:
                time.sleep(0)              # yield the GIL right after the read, on purpose,
                                            # to widen the window another thread can interleave in
            counter = temp + 1             # WRITE counter (based on the possibly-stale READ)

    threads = [threading.Thread(target=worker) for _ in range(NUM_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return counter


def run_safe():
    """Same threads, same interleaving points, but the critical section is now locked."""
    counter = 0
    lock = threading.Lock()

    def worker():
        nonlocal counter
        for i in range(INCREMENTS_PER_THREAD):
            with lock:
                temp = counter
                if i % 500 == 0:
                    time.sleep(0)
                counter = temp + 1

    threads = [threading.Thread(target=worker) for _ in range(NUM_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return counter


if __name__ == "__main__":
    print(f"Expected count every run: {EXPECTED}\n")

    print("=== WITHOUT threading.Lock (race condition) ===")
    for run in range(10):
        actual = run_unsafe()
        status = "OK" if actual == EXPECTED else "WRONG (lost updates)"
        print(f"run {run+1:2d}: counter = {actual:5d}  -> {status}")

    print("\n=== WITH threading.Lock (fixed) ===")
    for run in range(10):
        actual = run_safe()
        status = "OK" if actual == EXPECTED else "WRONG (lost updates)"
        print(f"run {run+1:2d}: counter = {actual:5d}  -> {status}")
