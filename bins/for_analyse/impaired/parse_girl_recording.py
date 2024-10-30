from pydub import AudioSegment
import os

def process_files_and_cut_audio(cased_file, aligned_file, output_file, audio_dir, output_audio_dir):
    # Read the transcript lines
    with open(cased_file, 'r', encoding='utf-8') as f:
        transcripts = f.readlines()
    
    # Read the aligned lines
    with open(aligned_file, 'r', encoding='utf-8') as f:
        aligned_data = f.readlines()
    
    # Ensure output audio directory exists
    os.makedirs(output_audio_dir, exist_ok=True)
    
    # Process and write the combined data
    with open(output_file, 'w', encoding='utf-8') as f:
        for transcript, aligned in zip(transcripts, aligned_data):
            # Strip whitespace and newline characters
            transcript = transcript.strip()
            aligned = aligned.strip()
            
            # Split the aligned data into parts
            parts = aligned.split()
            
            # Extract information
            segment_id = parts[0]
            audio_path = parts[1]
            start_time = float(parts[2])
            end_time = float(parts[3])
            
            # Load the audio file
            full_audio_path = os.path.join(audio_dir, audio_path)
            audio = AudioSegment.from_mp3(full_audio_path)
            
            # Cut the audio segment
            start_ms = int(start_time * 1000)
            end_ms = int(end_time * 1000)
            audio_segment = audio[start_ms:end_ms]
            
            # Generate new audio filename
            new_audio_filename = f"{segment_id}.wav"
            new_audio_path = os.path.join(output_audio_dir, new_audio_filename)
            
            # Export the audio segment
            audio_segment.export(new_audio_path, format="wav")
            
            # Write the combined line to the output file
            f.write(f"{new_audio_filename}|{transcript}\n")

# Usage
cased_file = '/project/tts/students/yining_ws/multi_lng/F5-TTS/dataset/impaired/impairedSpeech.DE.test.cased'
aligned_file = '/project/tts/students/yining_ws/multi_lng/F5-TTS/dataset/impaired/impairedSpeech.DE.test.seg.aligned'
output_file = '/project/tts/students/yining_ws/multi_lng/F5-TTS/dataset/impaired/text_audio_pair.txt'
audio_dir = '/project/tts/students/yining_ws/multi_lng/F5-TTS/dataset/impaired'
output_audio_dir = '/project/tts/students/yining_ws/multi_lng/F5-TTS/dataset/impaired/data_impairedSpeech_new/segments'

process_files_and_cut_audio(cased_file, aligned_file, output_file, audio_dir, output_audio_dir)