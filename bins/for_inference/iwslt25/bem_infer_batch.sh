MY_PATH="/project/tts/students/yining_ws/multi_lng/F5-TTS"
# export CUDA_VISIBLE_DEVICES=5
cd $MY_PATH
accelerate launch --gpu_ids=0,1 src/f5_tts/eval/eval_infer_batch.py \
--seed 0 \
--expname iwslt25/bigc_bem_finetune/E2TTS_Base_vocos_pinyin_bigc_bem \
--nfestep 32 \
--swaysampling -1 \
--infer_batch_size 12800 \
--cfg_strength 2 \
--speed 1 \
--ckptstep 50000 \
--testset iwslt25_bem \
--config /project/tts/students/yining_ws/multi_lng/F5-TTS/src/f5_tts/configs/iwslt25/E2TTS_Base_finetune_bigc_ben.yaml \
--input_file /project/tts/students/yining_ws/multi_lng/F5-TTS/inputs/iwslt25/60000_with_info.txt \
--output_dir outputs/iwslt25_augmented_withbatch_60000