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
import string
import fasttext
#while "/project/OML/vision/seyma/facedubbing/multilingual" in sys.path:
#   sys.path.remove("/project/OML/vision/seyma/facedubbing/multilingual")
#sys.path.remove("/project/OML/vision/seyma/facedubbing/schwabisch")
# sys.path.append("/project/tts/students/yining_ws/multi_lng/TTS")
# print(sys.path)
#import scipy.io.wavfile
#from TTS.config import load_config
if "/project/tts/students/yining_ws/multi_lng/TTS" not in sys.path:
    sys.path.append("/project/tts/students/yining_ws/multi_lng/TTS")
    
if "/project/tts/students/yining_ws/multi_lng/vits" not in sys.path:
    sys.path.append("/project/tts/students/yining_ws/multi_lng/vits")
    
if "/project/OML/vision/seyma/facedubbing/multilingual/TTS" in sys.path:
    sys.path.remove("/project/OML/vision/seyma/facedubbing/multilingual/TTS")

if "/project/OML/vision/seyma/facedubbing/multilingual" in sys.path:
    sys.path.remove("/project/OML/vision/seyma/facedubbing/multilingual")
    
print(sys.path)
from TTS.utils.synthesizer import Synthesizer
from TTS.tts.configs.vits_config import *
#from TTS.api import TTS
import requests

#from meta-ai import their 1000 languages VITS

from VitsSynthesizer import VitsSynthesizer

#print(sys.path)
host = sys.argv[1]  # 192.168.0.72
port = sys.argv[2]  # 5051

app = Flask(__name__)

root_dir = "/project/tts/students/yining_ws/multi_lng/TTS/checkpoints/"

model_en = VitsSynthesizer(
        lang="eng",
        ckpt_dir="/project/tts/students/yining_ws/multi_lng/vits/checkpoints/",
        use_cuda=True
    )

model_de = VitsSynthesizer(
        lang="deu",
        ckpt_dir="/project/tts/students/yining_ws/multi_lng/vits/checkpoints/",
        use_cuda=True
    )

model_fr = VitsSynthesizer(
        lang="fra",
        ckpt_dir="/project/tts/students/yining_ws/multi_lng/vits/checkpoints/",
        use_cuda=True
    )

model_es = VitsSynthesizer(
        lang="spa",
        ckpt_dir="/project/tts/students/yining_ws/multi_lng/vits/checkpoints/",
        use_cuda=True
    )


#model_ja = Synthesizer(
#       tts_checkpoint=root_dir+"jp/VitsProsody_japanese_phoneme_woemph_css10_dur20_unethead8-March-22-2024_08+12PM*/best_model.pth",
#        tts_config_path=root_dir+"jp/VitsProsody_japanese_phoneme_woemph_css10_dur20_unethead8-March-22-2024_08+12PM*/config.json",
#    )

model_ja = Synthesizer(
        tts_checkpoint="/project/tts/students/yining_ws/multi_lng/TTS/outputs_multi_lingual/phoneme_version/japanese_vits_retrained-July-02-2023_11+26PM-fd7f84d9/checkpoint_687000.pth",
        tts_config_path="/project/tts/students/yining_ws/multi_lng/TTS/outputs_multi_lingual/phoneme_version/japanese_vits_retrained-July-02-2023_11+26PM-fd7f84d9/config.json",
    )

model_zh= Synthesizer(
        tts_checkpoint=root_dir+"zh/chinese_retrain_vits_phwordblank_transformer_flow_posterior-November-08-2023_02+03PM*/best_model.pth",
        tts_config_path=root_dir+"zh/chinese_retrain_vits_phwordblank_transformer_flow_posterior-November-08-2023_02+03PM*/config.json",
    )

model_per = Synthesizer(
        tts_checkpoint=root_dir+"per/best_model_30824.pth",
        tts_config_path=root_dir+"per/config.json",
    )


#model_ita= Synthesizer(
#       tts_checkpoint=root_dir+"it/VitsProsody_ita_glow_attr6_woemphasis_withphoneme-April-08-2024_01+27PM*/best_model.pth",
#        tts_config_path=root_dir+"it/VitsProsody_ita_glow_attr6_woemphasis_withphoneme-April-08-2024_01+27PM*/config.json",
#    )

"""
#model_vc = TTS(model_name="voice_conversion_models/multilingual/vctk/freevc24", progress_bar=False, gpu=True)
"""
synthesizers = {
    "en_US": model_en,
    "de_DE": model_de,
    "fr_FR": model_fr,
    "es_ES": model_es,
    "ja_JP": model_ja,
    "zh_CN": model_zh,
    "per": model_per,
    #"ita": model_ita
}

# loading models from meta-mms
# arabic, vietnamese, thailandish, turkish, polish, russian, ukraian, korean, portulgish, hindi, bengali
for lang in ["ara", "vie", "tha", "tur", "pol", "rus", "ukr", "kor", "por", "hin", "ben"]:
    synthesizers[lang] = VitsSynthesizer(
        lang=lang, 
        ckpt_dir="/project/tts/students/yining_ws/multi_lng/vits/checkpoints/", 
        use_cuda=True
    )


# https://fasttext.cc/docs/en/language-identification.html
lang_id_model = fasttext.load_model('/project/tts/students/yining_ws/multi_lng/vits/checkpoints/lang_recognition/lid.176.bin')
lang_mapping = {"en":"en_US", "de":"de_DE",  "fr":"fr_FR",  "zh":"zh_CN",  "es":"es_ES",  "ja":"ja_JP",  "fa":"per", "ar":"ara",  "vi":"vie", \
        "th":"tha", "tr":"tur", "pl":"pol", "ru":"rus", "uk":"ukr", "ko":"kor", "pt":"por", "hi":"hin","bn":"ben"}

def pcm_s16le_to_array(pcm_s16le):
    audio_array = np.array(np.frombuffer(pcm_s16le, dtype=np.int16))
    audio_array = audio_array.astype(np.float32) / math.pow(2, 15)
    model_en.save_wav(audio_array, "input.wav")
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


#def vc(source_wav, target_wav, language):
   
    #waveform = model_multi.tts(speaker_wav=target_wav, reference_wav=source_wav, language_name=language)
    #model_multi.save_wav(waveform[0], "output_vc.wav")
    #return waveform


# corresponds to an asr_server "http://$host:$port/asr/infer/en,en" in StreamASR.py
# use None when no input- or output language should be specified
def tts_inference(language, text, pcm_s16le,len_scale):
    model = synthesizers[language]
    model.length_scale = len_scale

    if language == "es_ES":
        model.length_scale = 1.3#"2.0"
        
    if language == "fr_FR":
        model.length_scale = 1.3#"1.5"
    
    if language == "ja_JP":
        model.length_scale = "1.5"
        waveform = model.tts(text, language_name="ja_JP", speaker_name="jp_jsut")
    else:
        waveform = model.tts(text)

    
    with open("/project/OML/vision/seyma/facedubbing/webclient/new/exceptions.json") as exceptions_file: 
        
        exceptions = json.load(exceptions_file)
    
        words = text.split(" ")

        for word in words:
            w_no_punc = word.translate(str.maketrans("","", string.punctuation))
            if word.lower() in exceptions.keys() and exceptions[word.lower()]["lang"] == language:
                text = text.replace(word, exceptions[word.lower()]["text"])
                
            elif w_no_punc.lower() in exceptions.keys() and exceptions[w_no_punc.lower()]["lang"] == language:
                text = text.replace(w_no_punc, exceptions[w_no_punc.lower()]["text"])

    if language=="ja_JP":
        language_name="ja_JP"
        speaker_name="jp_jsut"
        #speaker_name=None
    elif language=="zh_CN":
        language_name="zh_CN"
        speaker_name="baker"
        # speaker_name=None
    elif language=="de_SW":
        language_name = "schwaebisch"
        speaker_name=None
    elif language=="fr_FR":
        language_name = "fr_FR"
        speaker_name=None
    elif language=="de_DE":
        language_name = "de_DE"
        speaker_name=None
    else:
        language_name=None
        speaker_name=None

    if language in ["en_US", "de_DE", "de_SW", "es_ES", "fr_FR", "zh_CN", "ja_JP", "per"]:
        waveform = model.tts(text, language_name=language_name, speaker_name=speaker_name)
    else:
        waveform = model.tts(text)

    if pcm_s16le:
        # TODO not safe here, the sampling rate might not be 22050 or 16000
        #model.output_sample_rate = 22050
        #model.save_wav(np.array(waveform[0]), "/export/data1/sakti/facedubbing/output.wav")      
        #waveform = list_to_pcm_s16le(waveform[0])
        np.save("/export/data1/sakti/facedubbing/output.npy", np.array(waveform[0]).astype(np.float16))

        target_wav = pcm_s16le_to_array(pcm_s16le)
        np.save("/export/data1/sakti/facedubbing/input.npy", target_wav.astype(np.float16))
        #model.output_sample_rate = 16000
        #model.save_wav(target_wav, "/export/data1/sakti/facedubbing/input.wav")
        print("Voice conversion starting")
        url = "http://i13hpc68:5054/tts/infer/vc"
        data = {"source_wav": "/export/data1/sakti/facedubbing/output.npy", "target_wav": "/export/data1/sakti/facedubbing/input.npy"}        
        vc_output = requests.post(url, files=data)
        #vc_output_2 = vc("/project/OML/vision/seyma/facedubbing/multilingual/output.wav", "/project/OML/vision/seyma/facedubbing/multilingual/input.wav", language)
        result = {"audio": vc_output.json()["vc_output"]}
        status = 200
        
    else:
        # model.save_wav(np.array(waveform[0]), "output.wav")
        waveform = list_to_pcm_s16le(waveform[0])
        result = {"audio": waveform}
        status = 200
        
    #return Response(send(waveform)), status
    
    return json.dumps(result), status


@app.route("/tts/infer/multi", methods=["POST"])
def multi_inference():
    #language: str = request.files.get("lang").read().decode("utf-8")
    text: str = request.files.get("seq").read().decode("utf-8")
    pcm_s16le = request.files.get("pcm_s16le")
    len_scale: float = request.files.get("len_scale")

    (language,), prob = lang_id_model.predict(text)
    language = lang_mapping[language.replace('__label__', '')]
    
    if pcm_s16le:
        pcm_s16le = pcm_s16le.read()
        print("audio: ", len(pcm_s16le))
        
    else:
        pcm_s16le = None
        
    if len_scale:
        len_scale = float(len_scale.read().decode("utf-8"))
    
    else:
        len_scale=1.0

    print("language: ", language)
    print("text: ", text)
    print("len_scale", len_scale)

    # calculate features corresponding to a torchaudio.load(filepath) call
    return tts_inference(language, text, pcm_s16le,len_scale)
    #model.tts_config.length_scale = len_scale
    

@app.route("/tts/infer/<language>", methods=["POST"])
def inference(language): 
    text: str = request.files.get("seq").read().decode("utf-8")
    pcm_s16le = request.files.get("pcm_s16le")
    len_scale: float = request.files.get("len_scale")
    
    if pcm_s16le:
        pcm_s16le = pcm_s16le.read()
        print("audio: ", len(pcm_s16le))
        
    else:
        pcm_s16le = None
        
    if len_scale:
        len_scale = float(len_scale.read().decode("utf-8"))
    
    else:
        len_scale=1.0

    print("language: ", language)
    print("text: ", text)
    print("len_scale", len_scale)

    # calculate features corresponding to a torchaudio.load(filepath) call
    return tts_inference(language, text, pcm_s16le,len_scale)

@app.route("/tts/version", methods=["POST"])
def version():
    # return a version number string (as first argument)
    return "1"#TODO, 200


app.run(host=host, port=port)
