#!/usr/bin/env python3
import argparse
import base64
import json
import re
import time
import numpy as np
import httpx
import requests
import subprocess
import tempfile
import os
import soundfile as sf

def synthesize_speech(text, speed, iteration_steps):
    tts_server = "http://192.168.0.64:5053/tts/infer/e2tts"
    timeout = None
    try:
        # print('#'*50)
        # print('TTS server: ', tts_server)
        # print('#'*50)
        # Sending a POST request using the requests library
        response = requests.post(
                tts_server,
                files={
                    # "pcm_s16le": audio,
                    "text": text,
                    # "lang": language_mapping.get(language),
                    "speed": str(speed),
                    "iteration_steps": iteration_steps
                },
                timeout=timeout
        )
        # Check if the response status code is not 200
        if response.status_code != 200:
            raise requests.ConnectionError(
                "HTTP return code of TTS model request is not equal to 200.")
    except requests.ConnectionError:
        print("ERROR in TTS model request, returning empty string.")
        return b''
    except requests.Timeout:
        print("TIMEOUT in TTS model request, returning empty string.")
        return b''
    else:
        audio_bytes = base64.b64decode(response.json()["audio"])
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        return audio_array

def main():
    text = "Kleidung macht Leute."
    speed = "1.0"
    iteration_steps = "10"
    start_time = time.time()
    audio_array = synthesize_speech(text, speed, iteration_steps)
    print(time.time()-start_time)
    if len(audio_array) > 0:
        wave_path = "/project/tts/students/yining_ws/multi_lng/F5-TTS/outputs/worker_output1.wav"
        final_sample_rate = 24000
        
        # Convert int16 to float32
        audio_float = audio_array.astype(np.float32) / 32768.0
        
        # Save as WAV file
        sf.write(wave_path, audio_float, final_sample_rate)
        print(f"Audio saved to {wave_path}")
    else:
        print("No audio data to save.")



if __name__ == "__main__":
    main()