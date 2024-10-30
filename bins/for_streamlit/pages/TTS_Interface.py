import streamlit as st
import os
import time
import string
from glob import glob
import torch
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
from googletrans import Translator
from numpy import random
import scipy.io.wavfile
import scipy.signal
import argparse
import base64
import json
import re
import numpy as np
import httpx
import requests
import subprocess
import tempfile
import soundfile as sf

import sys
#-----------add E2-TTS model here-----------------------------
path_to_add = '/project/tts/students/yining_ws/multi_lng/F5-TTS/src'
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
#-------------------------------------------------------------

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] =""
PROJECT =""
REGION =""

st.set_page_config(layout="wide", page_title="Multi-lingual TTS")

padding_top = 100

# st.markdown(
#     """
#     <style>
#     .reportview-container {
#         background: url("https://www.example.com/image.jpg");
#     }
#    </style>
#     """,
#     unsafe_allow_html=True
# )

st.markdown(f"""
    <style>
        .reportview-container .main .block-container{{
            padding-top: {padding_top}rem;
        }}
    </style>""",
    unsafe_allow_html=True,
)

# st.markdown("""
# <style>
#     [data-testid=stSidebar] {
#         background-color: white;
#     }
# </style>
# """, unsafe_allow_html=True)

st.write("## Text-to-Speech Interface")

model1_name = 'Zero-shot TTS trained on large-scaled data'
model2_name = 'Streaming'

st.sidebar.write("## Choose at least one model :gear:")
st.sidebar.write("## ")
model1 = st.sidebar.checkbox(f'{model1_name}')
model2 = st.sidebar.checkbox(f'{model2_name}')
# model3 = st.sidebar.checkbox(f'{model3_name}')
# model2 = st.sidebar.checkbox('Use the pitch-version.')

# @st.cache_resource()
# def LoadingTranslator():
#     translator = Translator()
#     return translator


# st.markdown('### ')
text = st.text_input("Please enter some text")

fairy_tale = st.selectbox("Select one fairy tale to introduce",
    ("0.None", "1.Cinderella", "2.Beauty and the Beast", "3.Sleeping Beauty", "4.Rapunzel")
)
fairy_tale_dict = {
    "1.Cinderella": """The story of Cinderella tells of a kind-hearted young woman who was treated cruelly by her stepmother and sisters, but, never the less, kept a humble attitude.
One day, the king decided to throw a ball and invited all the young maidens in the kingdom. While Cinderella’s sisters made her help them get ready for the ball, not once did they ask her if she would like to go with them.
Once they left, her Fairy Godmother appeared and helped Cinderella go to the ball with a bit of magic that would only last until midnight. At the ball, Cinderella caught the eye of the prince, as she was the most beautiful girl there, and they danced all night.
When midnight came, Cinderella had to leave the ball, and in her hurry, one of her glass slippers fell off her feet. The prince found this slipper and vowed to marry the girl who the slipper belonged to.
The prince went from house to house, looking for the girl who’s foot fit the slipper, and he reached Cinderella’s house. Though Cinderella’s stepsisters and stepmother tried to keep her from trying it on, the glass slipper was a perfect fit, and she was soon married to the prince and lived happily ever after.
This is a wonderful story that captures how keeping a humble attitude will reap its rewards.""",
    "2.Beauty and the Beast": """Originating in France, this is the story of Belle, a beautiful peasant girl who took the place of her father, when he was taken prisoner by a fierce beast.
While things were uncomfortable and frightening for Belle at first, she soon grew fond of the beast, as he had done nothing but treat her with kindness. When Belle found out her father was sick, she begged the beast to let her go to him and promised to return, but she was held up by the evil Gaston, a famous hunter from the village who wanted to marry Belle. When the village found out about the beast, they vowed to kill him and stormed his castle. Though he nearly died, he was saved and turned into a handsome prince because of Belle’s love for him. It turns out that he had been a prince who, along with his entire household, was cursed by a witch because he did not treat her with kindness. Belle and the Prince marry, and live a happy and peaceful life together.
From the prince’s curse, children can learn about the importance of being kind and that if they do not, they will suffer bad consequences. From Belle, we learn to value what is in a person’s heart, rather than their outward appearance.""",
    "3.Sleeping Beauty": """This is the story of Princess Aurora, the much-awaited daughter of the king and queen, who was cursed by an evil witch, to die by the prick from the spindle of a spinning wheel because her parents did not invite the fairy to her Christening.
Fortunately, one of the good fairies who had been invited to the Christening was able to help. Though the princess would still have to be pricked, she would not die, but sleep for a hundred years. She was blessed by the other good fairies, and so grew up to be a beautiful, kind and obedient young girl who was often called Briar Rose.
As predicted, on her sixteenth birthday, Aurora was pricked on her finger by a spinning wheel and fell into a deep sleep, along with every man, woman, child and animal in the castle.
A hundred years later, a young prince tried to get to the castle, in order to see the famous beauty that had been asleep for so long. When he found her, he was stunned by her beauty and leaned in for a kiss. This broke the curse, and soon everyone in the castle was awoken from their long, hundred year sleep. The prince and princess were married, and the kingdom was happy and peaceful once again.
Sleeping beauty teaches us that even though evil can sometimes interrupt our lives, when good intervenes, it can soften the blow and eventually, evil will be overcome.""",
    "4.Rapunzel": """A poor couple got themselves into big trouble when they stole fruit from their neighbour’s garden  The neighbour, who was a witch, found out about the theft and demanded that they give her their child when she was born, to which the couple accepted.
The young girl, named Rapunzel by the witch, grew up to be very beautiful, but was kept locked away in the tower by the wicked witch, from which there was no way in or out. When the witch wanted to go in and see her, she would say “Rapunzel, Rapunzel, let down your hair, so that I might climb the golden stair.”
One day, when Rapunzel was singing to pass the time, she happened to catch the attention of a young prince, who was so enchanted by her voice that he learned the secret of how to get to her. While Rapunzel was startled by him at first, they soon fell in love. It so happened that Rapunzel accidentally told the witch, “My, you are much heavier than my prince!” after which the witch, infuriated, chopped off her hair and threw her out into the wilderness. The prince was blinded by thorns and roamed the land, lamenting his beloved Rapunzel.
When they found each other again, the prince being lured by a beautiful voice, they cried for joy, and the tears which fell from Rapunzel’s eyes went into the prince’s, and cleansed them, enabling him to see again. The two lived together in peace for the rest of their lives.
The important thing to take away from this story is that one should never steal because it can have bad consequences, as in the case of Rapunzel’s parents, who lost their beautiful daughter because they were greedy and stole fruits."""
}

# st.markdown('### ')
in_lang = st.selectbox(
    "Select input language",
    ("1.English", "2.Spanish", "3.French", "4.Hochdeutsch", "5.Schwäbisch", "6.Chinese", "7.Japanese",
    "8.Arabic", "9.Vietnamese", "10.Thai", "11.Korean", "12.Filipino(to be continued)", "13.Russian", "14.Ukrainian",
    "15.Hindi", "16.Portuguese", "17.Italian", "18.Persian", "19.Turkish"),
)
values = ("1.English", "3.French", "4.Hochdeutsch", "6.Chinese")
#  "5.Schwäbisch",
# "7.Japanese",
#           "8.Arabic", "9.Vietnamese", "10.Thai", "11.Korean", "12.Filipino(to be continued)", "13.Russian", "14.Ukrainian",
#           "15.Hindi", "16.Portuguese", "17.Italian", "18.Persian", "19.Turkish")
out_lang = st.selectbox(
    "Select output language (same as input by default; otherwise Google Translator is used)",
    values,
    index=values.index(in_lang)
)

google_lang_dict = {"1.English": "en", "2.Spanish": "es", "3.French": "fr",
              "4.Hochdeutsch": "de", "5.Schwäbisch": "de", "6.Chinese": "zh-cn", "7.Japanese": "ja",
              "8.Arabic": "ar", "9.Vietnamese": "vi", "10.Thai": "th",
              "11.Korean": "ko", "12.Filipino(to be continued)":"fi", "13.Russian": "ru",
              "14.Ukrainian": "uk", "15.Hindi": "hi", "16.Portuguese": "pt", "17.Italian": "it", "18.Persian": "fa", \
              "19.Turkish": "tr"}

karlos_lang_dict = {"1.English": "en", "2.Spanish": "es", "3.French": "fr",
              "4.Hochdeutsch": "de", "5.Schwäbisch": "de", "6.Chinese": "zh", "7.Japanese": "ja",
              "8.Arabic": "ar", "9.Vietnamese": "vi", "10.Thai": "th",
              "11.Korean": "ko", "12.Filipino(to be continued)":"fi", "13.Russian": "ru",
              "14.Ukrainian": "uk", "15.Hindi": "hi", "16.Portuguese": "pt", "17.Italian": "it", "18.Persian": "fa", \
              "19.Turkish": "tr"}

yining_lang_dict = {"1.English": "en_US", "2.Spanish": "es_ES", "3.French": "fr_FR",
              "4.Hochdeutsch": "de_DE", "5.Schwäbisch": "de_SW", "6.Chinese": "zh_CN", "7.Japanese": "ja_JP",
              "8.Arabic": "ar_AR", "9.Vietnamese": "vie", "10.Thai": "tha",
              "11.Korean": "kor", "12.Filipino(to be continued)":"fil", "13.Russian": "rus",
              "14.Ukrainian": "ukr", "15.Hindi": "hin", "16.Portuguese": "por", "17.Italian": "ita", "18.Persian": "per",\
              "19.Turkish": "tur"}

input_language = karlos_lang_dict[in_lang]
out_tran_language = karlos_lang_dict[out_lang]

options = None
output_language = None
# if out_lang == "1.English":
#     options = ["Native American Female", "Native British Male"]
# elif out_lang == "2.Spanish":
#     options = ["Native Spanish Male"]
# elif out_lang == "3.French":
#     options = ["Native French Female", "Native French Male"]
# elif out_lang == "4.Hochdeutsch":
#     options = ["Native German Female", "Native German Male"]
# elif out_lang == "5.Schwäbisch":
#     options = ["Native German Female", "Native German Male"]
# elif out_lang == "6.Chinese":
#     options = ["Native Chinese Female"]
# elif out_lang == "7.Japanese":
#     options = ["Native Japanese Female", "Native Japanese Male"]
# elif out_lang == "8.Arabic":
#     options = ["Native Arabic Male"]
# output_language = yining_lang_dict[out_lang]
# st.markdown('### ')
out_speaker = st.selectbox(
    "Select your speaker",
    options=["Neutral speaker", "Prof. Waibel", "Barack Obama"],
)
# "Donald Trump", "Kamala Harris", "Joe Biden"

out_speaker_dict = {
    "Neutral speaker": {
        "ref_audio": "/project/tts/students/yining_ws/multi_lng/F5-TTS/src/f5_tts/infer/examples/basic/basic_ref_en.wav",
        "ref_text": "Some call me nature, others call me mother nature."
    },
    "Prof. Waibel": {
        "ref_audio":
        "/project/tts/students/yining_ws/Masterarbeit/data_prosody/alex/alex_wavs/Alex_1.wav",
        "ref_text": "It began with talk and it ended with talk."
    },
    "Barack Obama": {
        "ref_audio":
        "/project/tts/students/yining_ws/multi_lng/F5-TTS/src/f5_tts/infer/examples/custom/obama.wav",
        "ref_text": "Few countries would be more affected by warmer planet."
    },
    "Donald Trump": {
        "ref_audio": "",
        "ref_text": ""
    },
    "Kamala Harris": {
        "ref_audio": "",
        "ref_text": ""
    },
    "Joe Biden": {
        "ref_audio": "",
        "ref_text": ""
    }
}
default_speed_dict = {
    "Neutral speaker": 1.3,
    "Prof. Waibel": 1.4,
    "Barack Obama": 0.8
}

# st.markdown('### ')
speed_factor = st.slider("Choose speed factor", min_value=0.5, max_value=2.0, step=0.1, value=default_speed_dict[out_speaker])
max_chunk_size = st.slider("Choose chunk size", min_value=5, max_value=25, step=1, value=18)

# st.markdown('### ')
with st.expander("Click to expand"):
    nfe_steps = st.slider("<Optional>Choose infer steps", min_value=2, max_value=32, step=2, value=32)
    cfg_strength = st.slider("<Optional>Choose cfg_strength", min_value=0.0, max_value=5.0, step=0.1, value=2.0)
    sway_sampling_coef = st.slider("<Optional>Choose sway_sampling_coef", min_value=-2.0, max_value=2.0, step=0.1, value=-1.0)

    cross_fade_duration = st.slider("<Optional>Choose cross_fade_duration", min_value=0.0, max_value=3.0, step=0.1, value=0.2)
    fixed_duration = st.text_input("<Optional> You could define an expected audio duration of seconds.")

language_mapping = {"1.English":"eng", "4.Hochdeutsch": "deu", "3.French": "fra", "6.Chinese":"chn"}

def open_file(txt_file):
    text_lists = []
    wav_lists = []
    rand_list =random.randint(100, size=(5))
    print(rand_list)

    with open(txt_file, "r", encoding="utf-8") as ttf:
        for i, line in enumerate(ttf):
            if i in rand_list:
                wav_name, text, text1 = line.rstrip("\n").split("|")
                wav_name = '/export/data1/yliu/multi_lingual/wav/en_US/LJSpeech-1.1/wavs/'+wav_name+'.wav'
                text_lists.append(text)
                wav_lists.append(wav_name)

    return text_lists, wav_lists


def translate_karlos(text, lang):
    mt_server = "https://inference.isl.iar.kit.edu/predictions/mt-en,"
    timeout = None
    try:
        print('#'*50)
        print('MT server: ', mt_server+lang)
        print('#'*50)
        # Sending a POST request using the requests library
        response = requests.post(
                mt_server+lang,
                data={
                    "text": text
                },
                timeout=timeout
        )
        # Check if the response status code is not 200
        if response.status_code != 200:
            raise requests.ConnectionError(
                "HTTP return code of MT model request is not equal to 200.")
    except requests.ConnectionError:
        print("ERROR in MT model request, returning empty string.")
        return b''
    except requests.Timeout:
        print("TIMEOUT in MT model request, returning empty string.")
        return b''
    else:
        text_output = response.json()["hypo"]
        return text_output


def synthesize_speech(audio, refer_text, text, speed, lang, nfe_steps, cfg_strength, sway_sampling_coef, fixed_duration, cross_fade_duration, max_chunk_size, streaming="False"):
    tts_server = "http://192.168.0.62:5053/tts/infer/e2tts/"
    timeout = None
    try:
        print('#'*50)
        print('TTS server: ', tts_server+language_mapping.get(lang))
        print('#'*50)
        # Sending a POST request using the requests library
        response = requests.post(
                tts_server+language_mapping.get(lang),
                files={
                    "pcm_s16le": audio,
                    # "ref_audio": audio,
                    "ref_text": refer_text,
                    "seq": text,
                    # "lang": language_mapping.get(language),
                    "len_scale": str(speed),
                    "nfe_steps": str(nfe_steps),
                    "cfg_strength": str(cfg_strength),
                    "sway_sampling_coef": str(sway_sampling_coef),
                    "fix_duration": fixed_duration,
                    "cross_fade_duration": str(cross_fade_duration),
                    "max_chunk_size": str(max_chunk_size),
                    "streaming": str(streaming)
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
        text_output = response.json()["text"]
        return audio_array, text_output

def cal_str(ph_in):
    ph_str = []
    for i, ph in enumerate(ph_in):
        if ph==' ':
            ph_str.append('~')
        else:
            ph_str.append(ph)
        # for _ in range(int(dr[i])):
    #     ph_str+=ph
    # ph_str = ph_str.replace('','~')
    # print(ph_str)
    return ph_str


model1 = True
# st.write('### ')
if st.button("Synthesis :rocket:"):
    if text is '' and fairy_tale=="0.None":
        st.error("Please enter some text or choose one fairy tale to read.")
    elif not model1 and not model2:
        st.error("Please select at least one approach from the sidebar.")
    else:
        if fairy_tale!="0.None":
            text = fairy_tale_dict[fairy_tale]

        st.markdown("---")
        st.write(f'{input_language} Input:')
        st.write(f'*{text}*')
        if input_language!=out_tran_language:
            # translator = LoadingTranslator()
            text = translate_karlos(text, out_tran_language)
            # translation = translator.translate(text, src=input_language, dest=out_tran_language)
            # text = translation.text
            # text = text.replace(",", ", ")
            # st.write('## ')
            st.write(f'{out_tran_language} Translated Output:')
            st.write(f'*{text}*')
            st.markdown("---")

        if model1:
            if fixed_duration=="":
                fixed_duration=None
            else:
                fixed_duration = str(fixed_duration)
            #
            start_time = time.time()
            # print('#'*5)
            # print('ref_audio: ', out_speaker_dict[out_speaker]["ref_audio"])
            refer_audio = out_speaker_dict[out_speaker]["ref_audio"]
            refer_text = out_speaker_dict[out_speaker]["ref_text"]
            audio_array, text_output = synthesize_speech(refer_audio, refer_text, text, speed_factor, out_lang, nfe_steps, cfg_strength, sway_sampling_coef, fixed_duration, cross_fade_duration, max_chunk_size, streaming="False")
            process_time = time.time() - start_time
            audio_time = len(audio_array) / 24000
            # Create a formatted string with the information
            info_text = f"""
                Audio time: {audio_time:.2f} seconds
                Processing time: {process_time:.2f} seconds
                Real-time factor: {(process_time / audio_time):.2f}
            """
            # Display the information in a text area
            st.text_area("Audio Processing Information", info_text)
            # height=100)
            # print(f" > Audio time: {audio_time}")
            # print(f" > Processing time: {process_time}")
            # print(f" > Real-time factor: {process_time / audio_time}")
            # print(audio_array)
            # audio_file = open(f"temp/{result_origin1}", "rb")
            # audio_bytes = audio_file.read()
            st.markdown(f'### {model1_name}:')

            txt = cal_str(text_output[0][0])
            st.text_area('text input', txt)

            st.audio(audio_array, format="audio/wav", start_time=0, sample_rate=24000)
            # st.text_area('current random seed', randn_ins[0])
            # text_name = result_origin.replace(".wav", ".txt")

            # if os.path.isfile(f"temp/{text_name}"):
            #     txt_pre = open(f"temp/{text_name}", "r").read()
            #     st.text_area('previous duration', txt_pre)
            # st.markdown("---")
            # open(f"temp/{text_name}", "w").write(txt)
            #
            # durs_origin = torch.tensor(durs_origin).unsqueeze(0)
            # dur_name = result_origin1.replace(".wav", ".pt")
            # # print(dur_name)
            # torch.save(durs_origin, (f"temp/{dur_name}")
        if model2:
            if fixed_duration=="":
                fixed_duration=None
            else:
                fixed_duration = str(fixed_duration)
            #
            start_time = time.time()
            # print('#'*5)
            # print('ref_audio: ', out_speaker_dict[out_speaker]["ref_audio"])
            refer_audio = out_speaker_dict[out_speaker]["ref_audio"]
            refer_text = out_speaker_dict[out_speaker]["ref_text"]
            audio_array, text_output = synthesize_speech(refer_audio, refer_text, text, speed_factor, out_lang, nfe_steps, cfg_strength, sway_sampling_coef, fixed_duration, cross_fade_duration, max_chunk_size, streaming="True")
            process_time = time.time() - start_time
            audio_time = len(audio_array) / 24000
            # Create a formatted string with the information
            info_text = f"""
                Audio time: {audio_time:.2f} seconds
                Processing time: {process_time:.2f} seconds
                Real-time factor: {(process_time / audio_time):.2f}
            """
            # Display the information in a text area
            st.text_area("Audio Processing Information", info_text)
            st.markdown(f'### {model2_name}:')

            txt = cal_str(text_output[0][0])
            st.text_area('text input', txt)

            st.audio(audio_array, format="audio/wav", start_time=0, sample_rate=24000)

def remove_files(n, file_type):
    mp3_files = glob(f"temp/*{file_type}")
    if len(mp3_files) != 0:
        now = time.time()
        n_days = n * 86400
        for f in mp3_files:
            if os.stat(f).st_mtime < now - n_days:
                os.remove(f)
                print("Deleted ", f)

remove_files(1, 'wav')
remove_files(1, 'txt')
remove_files(1, 'pt')