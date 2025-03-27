MY_PATH="/project/tts/students/yining_ws/multi_lng/F5-TTS"
# export TORCH_NCCL_BLOCKING_WAIT=1
# # Use torch.distributed Error Handling:
# export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_DEBUG=INFO
# export NCCL_TIMEOUT=1200  # Increase timeout to 1200 seconds (or another suitable value)
cd $MY_PATH
# --main_process_port 0 
accelerate launch src/f5_tts/train/finetune_cli_from_config.py --config-name E2TTS_Base_finetune_bigc_ben.yaml