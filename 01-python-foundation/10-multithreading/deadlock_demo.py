import threading
import time

# --- BROKEN: locks acquired in opposite orders by two threads ---
lock_a = threading.Lock()
lock_b = threading.Lock()


def worker_1_bad():
    with lock_a:
        time.sleep(0.1)  # give worker_2 time to grab lock_b
        with lock_b:
            pass


def worker_2_bad():
    with lock_b:
        time.sleep(0.1)
        with lock_a:  # opposite acquisition order from worker_1_bad -> deadlock
            pass


t1 = threading.Thread(target=worker_1_bad, daemon=True)
t2 = threading.Thread(target=worker_2_bad, daemon=True)
t1.start()
t2.start()
t1.join(timeout=2)
t2.join(timeout=2)

if t1.is_alive() or t2.is_alive():
    print("DEADLOCK: threads did not finish within the 2s timeout "
          f"(t1 alive={t1.is_alive()}, t2 alive={t2.is_alive()})")
else:
    print("both threads finished (no deadlock this run)")

print()

# --- FIXED: fresh locks, both threads always acquire lock_c before lock_d ---
lock_c = threading.Lock()
lock_d = threading.Lock()


def worker_1_fixed():
    with lock_c:
        time.sleep(0.1)
        with lock_d:
            pass


def worker_2_fixed():
    with lock_c:   # same order as worker_1_fixed now, not reversed
        time.sleep(0.1)
        with lock_d:
            pass


t3 = threading.Thread(target=worker_1_fixed)
t4 = threading.Thread(target=worker_2_fixed)
t3.start()
t4.start()
t3.join(timeout=2)
t4.join(timeout=2)
print(f"fixed version: t3 alive={t3.is_alive()}, t4 alive={t4.is_alive()} -> both completed cleanly")
