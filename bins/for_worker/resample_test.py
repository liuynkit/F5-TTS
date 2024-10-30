import torch
import torchaudio
import librosa
from scipy.io.wavfile import write
import numpy as np
import base64
import math

def list_to_pcm_s16le(audio_list):
    audio_list= np.array(audio_list)
    audio_list = (audio_list * math.pow(2, 15)).astype(np.int16)
    audio_list = audio_list.tobytes()
    # audio_list = base64.b64encode(audio_list).decode("ascii")
    return audio_list

def pcm_s16le_to_array(pcm_s16le):
    audio_array = np.array(np.frombuffer(pcm_s16le, dtype=np.int16))
    audio_array = audio_array.astype(np.float32) / math.pow(2, 15)
    write("target_vc.wav", 22050, audio_array)
    return audio_array
#
source_wav = "/project/tts/students/yining_ws/multi_lng/TTS/data/wav_22050/SSB0005/SSB00050001.wav"
wav_16000, _ = librosa.load(source_wav, sr=16000)

wav_22050, _ = librosa.load(source_wav)
wav_22050 = list_to_pcm_s16le(wav_22050)
print(wav_22050)
wav_22050 = pcm_s16le_to_array(wav_22050)
wav_22050 = torch.from_numpy(wav_22050).unsqueeze(0)

print('wav_16000: ', wav_16000)
write("/project/tts/students/yining_ws/multi_lng/F5-TTS/bins/for_worker/output_1.wav", 16000, wav_16000)
#
target_sr = 22050
if target_sr != 16000:
    wav_tgt = torch.from_numpy(wav_22050)
    resampler = torchaudio.transforms.Resample(target_sr, 16000)
    wav_tgt = resampler(wav_tgt).numpy()
    print('wav_tgt: ', wav_tgt)
    write("/project/tts/students/yining_ws/multi_lng/F5-TTS/bins/for_worker/output_2.wav", 16000, wav_tgt)


