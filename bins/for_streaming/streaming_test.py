import threading
import time
from queue import Queue

class AudioChunkProcessor:
    def __init__(self, timeout=5):
        self.timeout = timeout  # Time to wait for the next chunk
        self.audio_queue = Queue()  # Queue to hold audio chunks
        self.last_received_time = time.time()  # Timestamp of the last received chunk
        self.timer_thread = threading.Thread(target=self._monitor_chunks, daemon=True)
        self.is_running = True  # Flag to control the timer thread
        self.timer_thread.start()

    def _monitor_chunks(self):
        """Monitor the queue and handle timing logic."""
        while self.is_running:
            time_since_last_chunk = time.time() - self.last_received_time
            
            if time_since_last_chunk >= self.timeout:
                # Timeout has elapsed without new audio
                print("No new audio received in the last 5 seconds.")
                self.last_received_time = time.time()  # Reset the timer
            time.sleep(0.1)  # Prevent CPU overuse, check every 100ms

    def add_audio_chunk(self, audio_chunk):
        """Receive a new audio chunk and reset the timer."""
        self.audio_queue.put(audio_chunk)
        self.last_received_time = time.time()
        print("Received a new audio chunk. Timer reset.")

    def stop(self):
        """Stop the monitoring thread."""
        self.is_running = False
        self.timer_thread.join()
        print("Stopped the audio processor.")

# Example Usage
processor = AudioChunkProcessor()

try:
    # Simulate receiving audio chunks
    time.sleep(2)  # Wait for 2 seconds
    processor.add_audio_chunk("chunk1")

    time.sleep(3)  # Wait for 3 seconds
    processor.add_audio_chunk("chunk2")

    time.sleep(6)  # Wait for 6 seconds (timeout should trigger)
    processor.add_audio_chunk("chunk3")

    time.sleep(1)  # Add another chunk quickly
    processor.add_audio_chunk("chunk4")
finally:
    processor.stop()  # Stop the processor cleanly
