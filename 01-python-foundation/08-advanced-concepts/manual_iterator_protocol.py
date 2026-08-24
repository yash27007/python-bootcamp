class CountUpTo:
    """Manual iterator protocol: __iter__ + __next__, no yield."""
    def __init__(self, limit):
        self.limit = limit
        self.current = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current >= self.limit:
            raise StopIteration
        self.current += 1
        return self.current


print("Manual __iter__/__next__ class:")
for n in CountUpTo(5):
    print(n)

print("\nSame thing with a generator function (yield):")
def count_up_to(limit):
    current = 0
    while current < limit:
        current += 1
        yield current

for n in count_up_to(5):
    print(n)

print("\nExhaustion failure mode: a generator can only be iterated once")
gen = count_up_to(3)
print("first pass:", list(gen))
print("second pass on the SAME generator object:", list(gen))
