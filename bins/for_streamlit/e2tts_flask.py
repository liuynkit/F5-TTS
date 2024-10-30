from flask import Flask, request, Response
import argparse
import codecs
import os
import re
from pathlib import Path
from importlib.resources import files

import math
import base64
import json
import numpy as np
import soundfile as sf
import tomli
from cached_path import cached_path
import requests
import torchaudio

import sys
path_to_add = '/project/tts/students/yining_ws/multi_lng/F5-TTS/src'
# Check if the path is already in sys.path
if path_to_add not in sys.path:
    sys.path.append(path_to_add)
    print(f"Added {path_to_add} to sys.path")
else:
    print(f"{path_to_add} is already in sys.path")

from f5_tts.model import DiT, UNetT
from f5_tts.infer.utils_infer import (
    load_vocoder,
    load_model,
    preprocess_ref_audio_text,
    infer_process,
    streaming_process,
    remove_silence_for_generated_wav,
    remove_silence_for_generated_wav_raw
)
from f5_tts.model.utils import (
    get_tokenizer,
    convert_char_to_pinyin,
)

host = sys.argv[1]  # 192.168.0.72
port = sys.argv[2]  # 5051

app = Flask(__name__)

# spectrogram_path = Path(output_dir) / "infer_cli_out.png"
vocos_local_path = "../checkpoints/charactr/vocos-mel-24khz"

vocoder = load_vocoder(is_local=False, local_path=vocos_local_path)
ema_models = {}
#
model = "E2-TTS"
ckpt_file = "/project/tts/students/yining_ws/multi_lng/F5-TTS/ckpts/Emilia_DE/model_396000.pt"
vocab_file = "/project/tts/students/yining_ws/multi_lng/F5-TTS/data/Emilia_ZH_EN_pinyin/vocab.txt"

# load models
if model == "F5-TTS":
    model_cls = DiT
    model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
    if ckpt_file == "":
        repo_name = "F5-TTS"
        exp_name = "F5TTS_Base"
        ckpt_step = 1200000
        ckpt_file = str(cached_path(f"hf://SWivid/{repo_name}/{exp_name}/model_{ckpt_step}.safetensors"))
        # ckpt_file = f"ckpts/{exp_name}/model_{ckpt_step}.pt"  # .pt | .safetensors; local path
elif model == "E2-TTS":
    model_cls = UNetT
    model_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4)
    if ckpt_file == "":
        repo_name = "E2-TTS"
        exp_name = "E2TTS_Base"
        ckpt_step = 1200000
        ckpt_file = str(cached_path(f"hf://SWivid/{repo_name}/{exp_name}/model_{ckpt_step}.safetensors"))
        # ckpt_file = f"ckpts/{exp_name}/model_{ckpt_step}.pt"  # .pt | .safetensors; local path

print(f"Using {model}...")
ema_models["deu"] = load_model(model_cls, model_cfg, ckpt_file, vocab_file)
ckpt_file = "/project/tts/students/yining_ws/multi_lng/F5-TTS/ckpts/Emilia_ZH_EN/model_1200000.pt"
ema_models["eng"] = load_model(model_cls, model_cfg, ckpt_file, vocab_file)
ema_models["chn"] = load_model(model_cls, model_cfg, ckpt_file, vocab_file)
ckpt_file = "/project/tts/students/yining_ws/multi_lng/F5-TTS/ckpts/Emilia_FR/model_322800.pt"
ema_models["fra"] = load_model(model_cls, model_cfg, ckpt_file, vocab_file)

def list_to_pcm_s16le(audio_list):
    audio_list= np.array(audio_list)
    audio_list = (audio_list * math.pow(2, 15)).astype(np.int16)
    audio_list = audio_list.tobytes()
    audio_list = base64.b64encode(audio_list).decode("ascii")
    return audio_list

# Read the .wav file once
# Save as NumPy array
en_ref_audio = "/project/tts/students/yining_ws/multi_lng/F5-TTS/src/f5_tts/infer/examples/basic/basic_ref_en.wav"
en_ref_audio, sr = torchaudio.load(en_ref_audio)
target_sample_rate = 24000
if sr != target_sample_rate:
    resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
    en_ref_audio = resampler(en_ref_audio)
# save as pcmse_l16
en_ref_audio_pcms16le = list_to_pcm_s16le(en_ref_audio)

text_list = ["Load tokenizer in previous."]
final_text_list = convert_char_to_pinyin(text_list)

def main_process(ref_audio, ref_text, text_gen, model_obj, remove_silence, speed, nfe_steps, cfg_strength, sway_sampling_coef, fix_duration, cross_fade_duration, max_chunk_size, sr, streaming=False):
    main_voice = {"ref_audio": ref_audio, "ref_text": ref_text}
    voices = {"main": main_voice}
    for voice in voices:
        voices[voice]["ref_audio"], voices[voice]["ref_text"] = preprocess_ref_audio_text(
            voices[voice]["ref_audio"], voices[voice]["ref_text"]
        )
        # print("Voice:", voice)
        # print("Ref_audio:", voices[voice]["ref_audio"])
        print("Ref_text:", voices[voice]["ref_text"])

    generated_audio_segments = []
    reg1 = r"(?=\[\w+\])"
    chunks = re.split(reg1, text_gen)
    reg2 = r"\[(\w+)\]"
    for text in chunks:
        match = re.match(reg2, text)
        if match:
            voice = match[1]
        else:
            print("No voice tag found, using main.")
            voice = "main"
        if voice not in voices:
            print(f"Voice {voice} not found, using main.")
            voice = "main"
        text = re.sub(reg2, "", text)
        gen_text = text.strip()
        ref_audio = voices[voice]["ref_audio"]
        ref_text = voices[voice]["ref_text"]
        # print(f"Voice: {voice}")
        if streaming:
            audio, final_sample_rate, spectragram, text_seq_list = streaming_process(
                ref_audio, ref_text, gen_text, model_obj, vocoder, 
                cross_fade_duration=cross_fade_duration,
                speed=speed, 
                nfe_step=nfe_steps,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
                fix_duration=fix_duration,
                max_chunk_size=max_chunk_size
            )
        else:
            audio, final_sample_rate, spectragram, text_seq_list = infer_process(
                ref_audio, ref_text, gen_text, model_obj, vocoder, 
                cross_fade_duration=cross_fade_duration,
                speed=speed, 
                nfe_step=nfe_steps,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
                fix_duration=fix_duration,
                max_chunk_size=max_chunk_size,
                sr=sr
            )
        generated_audio_segments.append(audio)

    # if generated_audio_segments:
    final_wave = np.concatenate(generated_audio_segments)

    return final_wave, text_seq_list


@app.route("/tts/infer/<language>", methods=["POST"])
def inference(language):
    gen_text: str = request.files.get("seq").read().decode("utf-8")

    sr=float(request.files.get("sr").read().decode("utf-8"))

    refer_audio = request.files.get("pcm_s16le")
    # print("refer_audio: ", refer_audio)
    refer_text = request.files.get("ref_text")
    # print("refer_text: ", refer_text)
    if refer_audio!=None:
        ref_audio=refer_audio.read()
        # print("ref_audio: ", ref_audio)
    if refer_text!=None:
        ref_text=str(refer_text.read().decode("utf-8"))
    if refer_text==None or refer_audio==None:
        ref_text="Some call me nature, others call me mother nature."
        ref_audio=en_ref_audio_pcms16le
    
    streaming = request.files.get("streaming")
    if streaming!=None:
        streaming=bool(streaming.read().decode("utf-8").lower() == "true")
    else:
        streaming=False
    # pcm_s16le = request.files.get("ref_pcm_s16le").read()
    speed = request.files.get("len_scale")
    if speed!=None:
        speed=1.0/float(speed.read().decode("utf-8"))
    else:
        speed=1.0
    
    nfe_steps = request.files.get("iteration_steps")
    if nfe_steps!=None:
        nfe_steps= int(nfe_steps.read().decode("utf-8"))
    else:
        nfe_steps=32
    
    cfg_strength = request.files.get("cfg_strength")
    if cfg_strength!=None:
        cfg_strength = float(cfg_strength.read().decode("utf-8")) 
    else:
        cfg_strength = 2.0 

    sway_sampling_coef = request.files.get("sway_sampling_coef")
    if sway_sampling_coef!=None:
        sway_sampling_coef = float(sway_sampling_coef.read().decode("utf-8"))
    else:
        sway_sampling_coef = -1.0

    cross_fade_duration = request.files.get("cross_fade_duration")
    if cross_fade_duration!=None:
        cross_fade_duration = float(cross_fade_duration.read().decode("utf-8"))
    else:
        cross_fade_duration = 0.2

    max_chunk_size = request.files.get("max_chunk_size")
    if max_chunk_size!=None:
        max_chunk_size = int(max_chunk_size.read().decode("utf-8"))
    else:
        max_chunk_size = 18

    fix_duration = request.files.get("fix_duration")
    if fix_duration!=None:
        fix_duration= int(fix_duration.read().decode("utf-8"))

    # ref_audio = "/project/tts/students/yining_ws/multi_lng/F5-TTS/dataset/impaired/data_impairedSpeech_new/segments/text_1-usr0227_Balto_0031394_0032000.wav"
    # ref_text = "Was kann man tun?"
    # ref_audio = "/project/tts/students/yining_ws/Masterarbeit/data_prosody/alex/alex_wavs/Alex_1.wav"
    # ref_text = "It began with talk and it ended with talk."
    # print('ref_audio: ', ref_audio)
    # print('ref_text: ', ref_text)

    remove_silence = False
    final_wave, text_seq_list = main_process(ref_audio, ref_text, gen_text, ema_models[language], remove_silence, speed, nfe_steps, cfg_strength, sway_sampling_coef, fix_duration, cross_fade_duration, max_chunk_size, sr, streaming=streaming)

    # waveform = remove_silence_for_generated_wav_raw(final_wave)
    waveform = list_to_pcm_s16le(final_wave)

    result = {"audio": waveform, "text": text_seq_list}
    status = 200

    return json.dumps(result), status

app.run(host=host, port=port)