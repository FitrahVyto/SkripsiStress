import threading
import time

def worker():
    print("Thread is starting...")
    time.sleep(2)
    print("Thread is finishing...")

# Create threads
threads = []
for i in range(5):
    t = threading.Thread(target=worker)
    threads.append(t)
    t.start()

# Wait for all threads to finish
for t in threads:
    t.join()

print("All threads have finished. Active threads:", threading.active_count())
