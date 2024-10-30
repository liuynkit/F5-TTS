export PATH="/home/yliu/miniconda3/envs/e2tts/bin:$PATH"
MY_PATH="/project/tts/students/yining_ws/multi_lng/F5-TTS/src/f5_tts/train"
# export TORCH_NCCL_BLOCKING_WAIT=1
# # Use torch.distributed Error Handling:
# export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_DEBUG=INFO
# export NCCL_TIMEOUT=1200  # Increase timeout to 1200 seconds (or another suitable value)

export CUDA_VISIBLE_DEVICES=1,2,5
cd $MY_PATH
# 38400/64 = 600  600*256/24000=6.4s for sample-duration in average
# 12800/64 = 200  2s in average
# 19200/32 = 600
accelerate launch --mixed_precision='fp16' finetune_cli.py \
--exp_name E2TTS_Base \
--learning_rate 1e-4 \
--dataset_name Emilia_ZH_EN_DE \
--batch_size_per_gpu 19200 \
--batch_size_type frame \
--grad_accumulation_steps 4 \
--max_grad_norm 1.0 \
--epochs 100 \
--max_samples 32 \
--tokenizer pinyin \
--tokenizer_path Emilia_ZH_EN \
--save_per_updates 500 \
--last_per_steps 600 \
--finetune False \
--log_samples False \
--logger wandb