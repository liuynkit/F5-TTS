import numpy as np
import soundfile as sf
import torchaudio
import time
import torch
import librosa

# Read the .wav file once
ref_audio = '/project/tts/students/yining_ws/multi_lng/F5-TTS/dataset/DE_B00000_S08971_W000052.mp3'

start = time.time()
audio, sr = torchaudio.load(ref_audio)
print("audio size:" , audio.size())
print("time to load .mp3", time.time()-start)
target_sample_rate = 24000
if sr != target_sample_rate:
    resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
    audio = resampler(audio)
print("audio size:" , audio.size())
# Save as NumPy array
np.save('/project/tts/students/yining_ws/multi_lng/F5-TTS/dataset/ref_audio4doering.npy', audio)

# Later, load the data quickly
start = time.time()
loaded_data = np.load('/project/tts/students/yining_ws/multi_lng/F5-TTS/dataset/ref_audio4doering.npy')
print('loaded_data', torch.from_numpy(loaded_data).size())
print("time to load .npy", time.time()-start)

wav_tgt, _ = librosa.load(ref_audio, sr=24000)
print(torch.from_numpy(wav_tgt).size())