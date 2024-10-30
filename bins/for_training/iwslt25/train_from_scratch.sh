export PATH="/home/yliu/miniconda3/envs/e2tts/bin:$PATH"
MY_PATH="/project/tts/students/yining_ws/multi_lng/F5-TTS"
cd $MY_PATH 
# export CUDA_MODULE_LOADING=LAZY
export NCCL_DEBUG=INFO
export PYTHONUNBUFFERED=1

accelerate launch src/f5_tts/train/train.py --config-name E2TTS_Base_train_bigc_ben.yaml