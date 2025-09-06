"""
Real World Example : Multithreading for I/O bound tasks
Scenario: Web Scraping

Web scraping opten involves making numberous network requests to fetch web pages. These taska re I/O bound because they spend a lot of 
time waiting for responses from servers. 
Multithreading can significantly improve the performance by allowing multiple webpages to be fetched concurrently.

We will use these three links

https://python.langchain.com/docs/tutorials/rag/
https://python.langchain.com/docs/introduction/
https://python.langchain.com/docs/how_to/

"""

import threading
import requests
from bs4 import BeautifulSoup

urls=[
    'https://python.langchain.com/docs/tutorials/rag/',
    'https://python.langchain.com/docs/introduction/',
    'https://python.langchain.com/docs/how_to/'
]

def fetch_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content,'html.parser')
    print(f"Fetched {len(soup.text)} characters from {url}")

threads = []

for url in urls:
    thread = threading.Thread(target=fetch_content, args=(url,))  # Added comma after url so that the whole url is passed
    threads.append(thread)
    thread.start()

for thread in threads:
    thread.join()

print("All web pages fetched")