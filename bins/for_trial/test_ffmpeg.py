# import pysbd
# import requests
# import base64
# import wave

# seq = "As you mentioned earlier, you love dogs. Dogs are highly social animals known for their loyalty and affection towards humans. They are often used as service animals, therapy dogs, and companions. With over 340 recognized breeds, dogs come in various shapes, sizes, and temperaments, making them a diverse and beloved species."
# seg = pysbd.Segmenter(language='en', clean=True)
# sens = seg.segment(seq)

# def synthesize_speech(audio, text, language=None, len_scale=1.0, ref_text=None):
#     try:
#         # Sending a POST request using the requests library
#         response = requests.post(
#                     "http://192.168.0.68:5057/tts/infer/multi",
#                     files={
#                         "pcm_s16le": audio,
#                         # "ref_text": ref_text,
#                         "seq": text,
#                         "len_scale": str(len_scale),
#                         # "sr": str(16000)
#                     },
#                     timeout=None
#         )
#         # Check if the response status code is not 200
#         if response.status_code != 200:
#             raise requests.ConnectionError("HTTP return code of TTS model request is not equal to 200.")
#     except requests.ConnectionError:
#         print("ERROR in TTS model request, returning empty string.")
#         return b''
#     except requests.Timeout:
#         print("TIMEOUT in TTS model request, returning empty string.")
#         return b''
#     else:
#         # Process the response
#         audio = base64.b64decode(response.json()["audio"])
#         while len(audio) >= 2 and audio[:2] == b'\x00\x00':
#             audio = audio[2:]
#         while audio[-2:] == b'\x00\x00':
#             audio = audio[:-2]
#         return audio

# audio_buffer = b''
# chunk_size = 4*16000*2
# chunk_counter = 0
# audio_saved_buffer_list = []

# for sen in sens:
#     pcm_s16le = synthesize_speech(bytes(), sen, len_scale=1.0)
#     print(pcm_s16le)
#     audio_buffer += pcm_s16le
#     print("adding segments...")
#     if len(audio_buffer) >= chunk_size:
#         while len(audio_buffer) >= chunk_size:
#             print("buffer limit reached...")
#             chunk = audio_buffer[:chunk_size]
#             audio_buffer = audio_buffer[chunk_size:]
#             chunk_counter += 1
#             audio_saved_buffer_list.append(chunk)

# chunk = audio_buffer
# # audio_buffer = b''
# chunk_counter += 1
# audio_saved_buffer_list.append(chunk)

# def pcm_to_wav(pcm_data, output_filename):
#     # Set up WAV parameters
#     num_channels = 1  # Mono
#     sample_width = 2  # 16-bit
#     sample_rate = 16000 

#     # Create WAV file
#     with wave.open(output_filename, 'wb') as wav_file:
#         wav_file.setnchannels(num_channels)
#         wav_file.setsampwidth(sample_width)
#         wav_file.setframerate(sample_rate)
#         wav_file.writeframes(pcm_data)

# def chunk_to_wav(pcm_chunks):
#     # Convert each PCM chunk to WAV
#     for index, pcm_chunk in enumerate(pcm_chunks):
#         output_filename = f"/project/tts/students/yining_ws/multi_lng/F5-TTS/outputs/test/segmentation_debug_{index}.wav"
#         pcm_to_wav(pcm_chunk, output_filename)

# chunk_to_wav(audio_saved_buffer_list)

import wave
import os

def combine_wav_chunks(input_directory, output_filename):
    
    # Get all WAV files in the input directory
    wav_files = [f for f in os.listdir(input_directory) if f.endswith('.wav')]
    wav_files.sort()  # Ensure files are in the correct order

    # Read the first file to get audio parameters
    with wave.open(os.path.join(input_directory, wav_files[0]), 'rb') as first_wav:
        params = first_wav.getparams()

    # Open the output file
    with wave.open(output_filename, 'wb') as output_wav:
        output_wav.setparams(params)

        # Iterate through all input files
        for wav_file in wav_files:
            with wave.open(os.path.join(input_directory, wav_file), 'rb') as wav:
                output_wav.writeframes(wav.readframes(wav.getnframes()))

    print(f"Combined audio saved as {output_filename}")

# Usage
input_directory = "/project/tts/students/yining_ws/multi_lng/F5-TTS/outputs/test"
output_filename = "/project/tts/students/yining_ws/multi_lng/F5-TTS/outputs/combined_audio.wav"
combine_wav_chunks(input_directory, output_filename)
