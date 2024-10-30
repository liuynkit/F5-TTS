from flask import Flask, request, Response
import torch
import numpy as np
import math
import sys
import json
import threading
import queue
import uuid
import base64
import scipy.io.wavfile
import requests
import string
import torchaudio

path_to_add = '/project/tts/students/yining_ws/TTS'
# Check if the path is already in sys.path
if path_to_add not in sys.path:
    sys.path.append(path_to_add)
    print(f"Added {path_to_add} to sys.path")
else:
    print(f"{path_to_add} is already in sys.path")

from TTS.api import TTS

# init model
model = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

host = sys.argv[1]  # 192.168.0.72
port = sys.argv[2]  # 5051

app = Flask(__name__)

# init speaker
speaker = {
    "speaker_embedding": tts.synthesizer.tts_model.speaker_manager.speakers["Claribel Dervla"][
        "speaker_embedding"
    ]
    .cpu()
    .squeeze()
    .half()
    .tolist(),
    "gpt_cond_latent": tts.synthesizer.tts_model.speaker_manager.speakers["Claribel Dervla"][
        "gpt_cond_latent"
    ]
    .cpu()
    .squeeze()
    .half()
    .tolist(),
}


def pcm_s16le_to_array(pcm_s16le):
    audio_array = np.array(np.frombuffer(pcm_s16le, dtype=np.int16))
    audio_array = audio_array.astype(np.float32) / math.pow(2, 15)
    print(audio_array.shape)
    torchaudio.save("input.wav", torch.Tensor(audio_array).unsqueeze(0), 16000)
    return audio_array

def list_to_pcm_s16le(audio_list):
    audio_list= np.array(audio_list)
    audio_list = (audio_list * math.pow(2, 15)).astype(np.int16)
    audio_list = audio_list.tobytes()
    audio_list = base64.b64encode(audio_list).decode("ascii")
    return audio_list

def send(result):
    chunk_size = 1024*1024
    chunks = [result[i:i+chunk_size] for i in range(0, len(result), chunk_size)]
    for chunk in chunks:

        yield json.dumps({"audio": chunk})


def stream_tts(text, save_path):
    speaker_embedding = torch.tensor(speaker["speaker_embedding"],).unsqueeze(0).unsqueeze(-1)
    gpt_cond_latent = torch.tensor(speaker["gpt_cond_latent"]).reshape((-1, 1024)).unsqueeze(0)
    #text="[en]Madam President, we are not simply discussing domestic problems in the European Parliament Madam President, we are not simply discussing domestic problems in the European Parliament Madam President, we are not simply discussing domestic problems in the European Parliament Madam President, we are not simply discussing domestic problems in the European Parliament Madam President, we are not simply discussing domestic problems in the European Parliament Madam President, we are not simply discussing domestic problems in the European Parliament Madam President, we are not simply discussing domestic problems in the European Parliament Madam President, we are not simply discussing domestic problems in the European Parliament Madam President, we are not simply discussing domestic problems in the European Parliament Madam President, we are not simply discussing domestic problems in the European Parliament Madam President, we are not simply discussing domestic problems in the European Parliament Madam President, we are not simply discussing domestic problems in the European Parliament. [de]Die europäischen Werte werden in der neuen ungarischen Verfassung in Frage gestellt."
    language="en"

    streamer = model.synthesizer.tts_model.inference_stream(
            text,
            language,
            gpt_cond_latent,
            speaker_embedding,
            stream_chunk_size=150,
            enable_text_splitting=True,
        )

    wavs = []
    lens = []
    i = 0
    start = 0
    end = 0
    for num_sen, chunk in streamer:
        if num_sen > i:
            lens.append((start, end))
            wavs += [0] * 10000
            start = len(wavs)
            end = len(wavs)
            i += 1

        waveform = list(chunk.cpu().numpy())
        end += len(waveform)
        wavs += waveform

    lens.append((start, end))
    tts.synthesizer.save_wav(wavs, path=save_path)
    
    return lens


# corresponds to an asr_server "http://$host:$port/asr/infer/en,en" in StreamASR.py
# use None when no input- or output language should be specified
@app.route("/tts/infer/multi", methods=["POST"])
def inference():
    language: str = request.files.get("lang").read().decode("utf-8")
    text: str = request.files.get("seq").read().decode("utf-8")
    pcm_s16le = request.files.get("pcm_s16le").read()
    len_scale: float = float(request.files.get("len_scale").read().decode("utf-8"))

    print("language: ", language)
    print("text: ", text)
    print("audio: ", len(pcm_s16le))

    # calculate features corresponding to a torchaudio.load(filepath) call

    #model.tts_config.length_scale = len_scale
    #model.tts_config.use_speaker_embedding = False
    with open("/project/OML/vision/seyma/facedubbing/webclient/new/exceptions.json") as exceptions_file: 
        
        exceptions = json.load(exceptions_file)
    
        words = text.split(" ")

        for word in words:
            w_no_punc = word.translate(str.maketrans("","", string.punctuation))
            if word.lower() in exceptions.keys() and exceptions[word.lower()]["lang"] == language:
                text = text.replace(word, exceptions[word.lower()]["text"])
                
            elif w_no_punc.lower() in exceptions.keys() and exceptions[w_no_punc.lower()]["lang"] == language:
                text = text.replace(w_no_punc, exceptions[w_no_punc.lower()]["text"])
                
    if len(pcm_s16le) > 0:
        audio_array = pcm_s16le_to_array(pcm_s16le)
       
        print("TTS with voice conversion starting")
        #waveform = model.tts(text, len_scale=len_scale, speaker_wav="input.wav", language_name=language)
        # generate speech by cloning a voice using default settings
        model.length_scale = len_scale
        model.tts_to_file(text=text,
                    file_path="output.wav",
                    speaker_wav=["input.wav"],
                    language=language[:2],
                    split_sentences=True
                    )

    audio, sr = torchaudio.load("output.wav")
    audio=torchaudio.functional.resample(audio, sr, 16000)
    output = list_to_pcm_s16le(audio)
    
    result = {"audio": output}
    
    #model.save_wav(waveform[0], "output.wav")
    #waveform = list_to_pcm_s16le(waveform[0])
    #result = {"audio": waveform}

    status = 200

    #return Response(send(waveform)), status
    
    return json.dumps(result), status


# called during automatic evaluation of the pipeline to store worker information
@app.route("/tts/version", methods=["POST"])
def version():
    # return a version number string (as first argument)
    return "1"#TODO, 200


app.run(host=host, port=port)
