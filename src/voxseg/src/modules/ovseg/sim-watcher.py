import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from playground import Lightweight
import os

# Define the directory to watch and the file name to look for
directory = '/home/pcgta/Documents/eth/wild_visual_navigation/wild_visual_navigation_orbit'
file_name = 'construction.jpg'

predictor = Lightweight()

# Define a custom event handler
class FileEventHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.src_path.endswith(file_name):
            print('DETECTED')
            start_time = time.time()
            path = os.path.join(directory, file_name)
            predictor.run([path], ['watermark', 'other'])
            print(f'Total time: {time.time() - start_time}')

# Create an observer and attach the event handler
observer = Observer()
observer.schedule(FileEventHandler(), directory, recursive=False)

# Start the observer
print('STARTING OBSERVER')
observer.start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    observer.stop()

# Cleanup
observer.join()
