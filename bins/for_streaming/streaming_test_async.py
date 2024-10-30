import asyncio
from datetime import datetime, timedelta

class AsyncAudioProcessor:
    def __init__(self, timeout=5):
        self.timeout = timeout  # Time in seconds to wait for the next audio chunk
        self.last_received_time = datetime.now()  # Last time an audio chunk was received
        self.audio_queue = asyncio.Queue()  # Queue to hold audio chunks
        self.is_running = True

    async def monitor_chunks(self):
        """Monitor the queue and handle timing logic asynchronously."""
        while self.is_running:
            now = datetime.now()
            time_since_last_chunk = (now - self.last_received_time).total_seconds()
            
            # Check if the timeout period has passed
            if time_since_last_chunk >= self.timeout:
                print("No new audio received in the last 5 seconds.")
                self.last_received_time = datetime.now()  # Reset the timer
            
            # Sleep briefly to yield control
            await asyncio.sleep(0.1)  # Check every 100ms

    async def add_audio_chunk(self, audio_chunk):
        """Receive a new audio chunk and reset the timer asynchronously."""
        await self.audio_queue.put(audio_chunk)
        self.last_received_time = datetime.now()
        print(f"Received a new audio chunk at {self.last_received_time}. Timer reset.")

    async def stop(self):
        """Stop the monitoring loop."""
        self.is_running = False
        print("Stopped the audio processor.")

# Example Usage
async def main():
    processor = AsyncAudioProcessor()

    # Start the monitoring task
    monitor_task = asyncio.create_task(processor.monitor_chunks())

    # Simulate receiving audio chunks with delays
    await asyncio.sleep(2)
    await processor.add_audio_chunk("chunk1")

    await asyncio.sleep(3)
    await processor.add_audio_chunk("chunk2")

    await asyncio.sleep(6)  # This will trigger the timeout message
    await processor.add_audio_chunk("chunk3")

    await asyncio.sleep(1)
    await processor.add_audio_chunk("chunk4")

    # Stop the monitoring task
    await processor.stop()
    await monitor_task  # Wait for the monitor task to finish

# Run the async main function
asyncio.run(main())
