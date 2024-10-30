export PATH="/home/yliu/miniconda3/envs/e2tts/bin:$PATH"
MY_PATH="/project/tts/students/yining_ws/multi_lng/F5-TTS/src"
export CUDA_VISIBLE_DEVICES=0
cd $MY_PATH
python f5_tts/infer/infer_cli.py \
--model E2-TTS \
--ckpt_file /project/tts/students/yining_ws/multi_lng/F5-TTS/ckpts/Emilia_ZH_EN/model_1200000.pt \
--vocab_file /project/tts/students/yining_ws/multi_lng/F5-TTS/data/Emilia_ZH_EN_pinyin/vocab.txt \
--ref_audio /project/tts/students/yining_ws/multi_lng/F5-TTS/src/f5_tts/infer/examples/basic/basic_ref_en.wav \
--ref_text "Some call me nature, others call me mother nature." \
--gen_text "I don't really care what you call me. I've been a silent spectator, watching species evolve, empires rise and fall. But always remember, I am mighty and enduring." \
--output_dir /project/tts/students/yining_ws/multi_lng/F5-TTS/outputs/EN
