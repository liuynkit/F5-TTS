export PATH="/home/yliu/miniconda3/envs/e2tts/bin:$PATH"
MY_PATH="/project/tts/students/yining_ws/multi_lng/F5-TTS"
cd /project/tts/students/yining_ws/multi_lng/F5-TTS/outputs/streamlit
streamlit run $MY_PATH/bins/for_streamlit/Home.py --server.port 8080