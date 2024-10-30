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
    remove_silence_for_generated_wav,
)

host = sys.argv[1]  # 192.168.0.72
port = sys.argv[2]  # 5051

app = Flask(__name__)

# spectrogram_path = Path(output_dir) / "infer_cli_out.png"
vocos_local_path = "../checkpoints/charactr/vocos-mel-24khz"

vocoder = load_vocoder(is_local=False, local_path=vocos_local_path)

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
ema_model = load_model(model_cls, model_cfg, ckpt_file, vocab_file)

def main_process(ref_audio, ref_text, text_gen, model_obj, remove_silence, speed, nfe_steps):
    main_voice = {"ref_audio": ref_audio, "ref_text": ref_text}
    voices = {"main": main_voice}
    for voice in voices:
        voices[voice]["ref_audio"], voices[voice]["ref_text"] = preprocess_ref_audio_text(
            voices[voice]["ref_audio"], voices[voice]["ref_text"]
        )
        print("Voice:", voice)
        print("Ref_audio:", voices[voice]["ref_audio"])
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
        print(f"Voice: {voice}")
        audio, final_sample_rate, spectrogram, _ = infer_process(
            ref_audio, ref_text, gen_text, model_obj, vocoder, speed=speed, nfe_step=nfe_steps
        )
        generated_audio_segments.append(audio)

    # if generated_audio_segments:
    final_wave = np.concatenate(generated_audio_segments)

    return final_wave

def list_to_pcm_s16le(audio_list):
    audio_list= np.array(audio_list)
    audio_list = (audio_list * math.pow(2, 15)).astype(np.int16)
    audio_list = audio_list.tobytes()
    audio_list = base64.b64encode(audio_list).decode("ascii")
    return audio_list

@app.route("/tts/infer/e2tts", methods=["POST"])
def inference():
    # language: str = request.files.get("lang").read().decode("utf-8")
    # ref_text: str = request.files.get("ref_text").read().decode("utf-8")
    gen_text: str = request.files.get("text").read().decode("utf-8")
    # pcm_s16le = request.files.get("ref_pcm_s16le").read()
    speed: float = float(
        request.files.get("speed").read().decode("utf-8"))

    nfe_steps = request.files.get("iteration_steps")
    if nfe_steps!=None:
        nfe_steps= int(nfe_steps.read().decode("utf-8"))
    else:
        nfe_steps=32
    # ref_audio = "/project/tts/students/yining_ws/multi_lng/F5-TTS/dataset/impaired/data_impairedSpeech_new/segments/text_1-usr0227_Balto_0031394_0032000.wav"
    # ref_text = "Was kann man tun?"
    ref_audio = '/project/tts/students/yining_ws/multi_lng/F5-TTS/dataset/ref_audio4doering.npy'
    ref_text = 'Können Sie mir den richtigen Weg zeigen?'
    remove_silence = False
    final_wave = main_process(ref_audio, ref_text, gen_text, ema_model, remove_silence, speed, nfe_steps)

    waveform = list_to_pcm_s16le(final_wave)
    result = {"audio": waveform}
    status = 200

    return json.dumps(result), status

app.run(host=host, port=port)