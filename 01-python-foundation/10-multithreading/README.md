### Theory

### 1. Program
A **program** is a set of instructions written in a programming language that tells a computer what to do. Think of it as a recipe: it lists the steps needed to complete a task, but by itself, it doesn't do anything until you actually start it. 

**Examples of programs:**
- Web browsers (Chrome, Firefox)
- Games (Minecraft, Solitaire)
- Text editors (Notepad, VS Code)
- Calculators

### 2. Process

A **process** is what you get when a program is running (i.e., an instance of a program being executed). When you open a program, the computer loads it into memory and starts executing its instructions. Each process is independent and has its own memory and resources.

**Example:** If you open two web browsers, each one runs as a separate process with its own memory space.

**Resources required by a process:**

| Resource | Description |
|----------|-------------|
| Code Segment | Contains the executable instructions of the program |
| Data Segment | Stores global and static variables |
| Heap | Dynamic memory allocation area |
| Stack | Stores local variables and function call information |
| Registers | CPU storage locations for immediate data access |

### 3. Threads

A **thread** is a smaller unit of a process (i.e., it's a unit of execution within a process). A process can have one or more threads, and each thread can run parts of the program simultaneously. Threads share the same memory within a process, which makes it easier for them to communicate with each other.

**Example:** In a web browser:
- One thread might handle loading a web page
- Another thread keeps the user interface responsive
- A third thread might handle downloads

**Another example:** You open MS Paint (this is a process):
- You can create a thread to draw a box
- You can create another thread to draw a circle
- Both threads use the resources allocated for the MS Paint process

**Key difference:** Processes are isolated from each other, while threads within the same process share memory and resources.

---

## Python Multithreading

Python provides the `threading` module to create and manage threads. Multithreading is useful for I/O-bound tasks (like downloading files, reading/writing to disk, or network operations) but is less effective for CPU-bound tasks due to the Global Interpreter Lock (GIL).

### When to Use Multithreading
- When you need to perform multiple I/O operations concurrently (e.g., web scraping, file downloads)
- When you want to keep a user interface responsive while doing background work

### Basic Example: Creating Threads

```python
import threading
import time

def print_numbers():
	for i in range(5):
		print(f"Number: {i}")
		time.sleep(1)

def print_letters():
	for letter in 'abcde':
		print(f"Letter: {letter}")
		time.sleep(1)

# Create threads
t1 = threading.Thread(target=print_numbers)
t2 = threading.Thread(target=print_letters)

# Start threads
t1.start()
t2.start()

# Wait for both threads to finish
t1.join()
t2.join()
print("Done!")
```

**Output:**
```
Number: 0
Letter: a
Number: 1
Letter: b
...
Done!
```

### Example: Downloading Multiple URLs Concurrently

```python
import threading
import requests

def download_url(url):
	resp = requests.get(url)
	print(f"Downloaded {url}: {len(resp.content)} bytes")

urls = [
	'https://www.example.com',
	'https://www.python.org',
	'https://www.github.com',
]

threads = []
for url in urls:
	t = threading.Thread(target=download_url, args=(url,))
	t.start()
	threads.append(t)

for t in threads:
	t.join()
print("All downloads complete.")
```

### Best Practices
- Use `threading.Thread` for I/O-bound tasks.
- Use `threading.Lock` to prevent race conditions when threads share data.
- For CPU-bound tasks, consider using `multiprocessing` instead.
- Always call `join()` on threads to ensure they finish before your program exits.

### Resources
- [Python threading documentation](https://docs.python.org/3/library/threading.html)
- [Real Python: Threading in Python](https://realpython.com/intro-to-python-threading/)