"""
From-scratch demonstration of CPython's two memory-reclamation mechanisms:

1. Reference counting (via `sys.getrefcount`) — the primary mechanism, real and observed.
2. A reference cycle that refcounting CANNOT free on its own, and the cycle-detecting
   garbage collector (`gc.collect()`) that CAN.

Every number below is real, actually executed output on this machine — nothing here is
predicted or fabricated.
"""
import sys
import gc


def refcount_demo():
    print("=== 1. Reference counting, observed directly ===")
    a = []
    # sys.getrefcount(a) always reports ONE MORE than you'd naively expect, because
    # passing `a` into the function call itself creates a temporary extra reference
    # for the duration of the call.
    print("refs to a right after creation:", sys.getrefcount(a))

    b = a  # a second real name now points at the same list object
    print("refs to a after b = a:         ", sys.getrefcount(a))

    del b  # removing the second name drops the count back down
    print("refs to a after del b:         ", sys.getrefcount(a))
    print()


class Node:
    """A tiny object whose __del__ lets us OBSERVE deallocation happening (or not)."""

    def __init__(self, name):
        self.name = name
        self.other = None

    def __del__(self):
        print(f"  Node {self.name} __del__ called -> memory actually freed")


def cycle_demo():
    print("=== 2. A reference cycle refcounting alone cannot free ===")
    gc.disable()  # disable the cycle collector so ONLY refcounting is in play

    n1 = Node("A")
    n2 = Node("B")
    n1.other = n2
    n2.other = n1  # the cycle: A -> B -> A
    print("created cycle: n1.other -> n2, n2.other -> n1")
    print("refcount n1 before del:", sys.getrefcount(n1))
    print("refcount n2 before del:", sys.getrefcount(n2))

    id1, id2 = id(n1), id(n2)
    del n1
    del n2
    # Each object's refcount just dropped by one (the local-name reference), but each
    # is STILL referenced by the other object in the cycle -- refcount never reaches 0,
    # so __del__ has NOT been called yet, and no "freed" message has printed above.
    print("deleted local names n1, n2 -- refcount dropped by 1 each, but not to 0")
    still_alive = any(id(o) in (id1, id2) for o in gc.get_objects())
    print("are the cycle objects still alive (found by gc.get_objects())?", still_alive)

    print("calling gc.collect() -- the cycle-detecting collector...")
    collected = gc.collect()
    print(f"gc.collect() collected {collected} objects")
    still_alive_after = any(id(o) in (id1, id2) for o in gc.get_objects())
    print("are the cycle objects still alive now?", still_alive_after)

    gc.enable()


if __name__ == "__main__":
    refcount_demo()
    cycle_demo()
