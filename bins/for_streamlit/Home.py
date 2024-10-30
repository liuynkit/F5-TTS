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
# from TTS.tts.utils.helpers import average_over_durations
# from googletrans import Translator
# from time import sleep
# from stqdm import stqdm
# try:
#     os.mkdir("temp")
# except:
#     pass

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] =""
PROJECT =""
REGION =""

st.set_page_config(layout="wide", page_title="Multi-lingual TTS")

padding_top = 0

st.markdown(f"""
    <style>
        .reportview-container .main .block-container{{
            padding-top: {padding_top}rem;
        }}
    </style>""",
    unsafe_allow_html=True,
)

# st.write("## Yining's TTS Diary")
# st.write("# ")
# st.write("# ")
# st.write("""
#         This is a website recording Yining's running TTS :sound: projects...as a TTS diary:smile:.\n
#          """)
# model2 = st.sidebar.checkbox('Use the pitch-version.')