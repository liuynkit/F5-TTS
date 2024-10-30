MY_PATH="/project/tts/students/yining_ws/multi_lng/F5-TTS/src"
export CUDA_VISIBLE_DEVICES=5
cd $MY_PATH
python f5_tts/infer/infer_cli_from_dataset.py \
--model E2-TTS \
--model_cfg /project/tts/students/yining_ws/multi_lng/F5-TTS/src/f5_tts/configs/iwslt25/E2TTS_Base_train_bigc_ben.yaml \
--ckpt_file /project/tts/students/yining_ws/multi_lng/F5-TTS/ckpts/iwslt25/bigc_bem_finetune/E2TTS_Base_vocos_pinyin_bigc_bem/checkpoints/model_last.pt \
--vocab_file /project/tts/students/yining_ws/multi_lng/F5-TTS/data/bigc_bem_pinyin/vocab.txt \
--ref_audio /project/OML/zli/iwslt2024/low_resourced_track/data/bem_eng/training/bigc/bigc/data/bem/audio/26_8819_0_261_01_211112-010515_bem_8f1_elicit_0.wav \
--ref_text "Abalumendo babili nabeminina bale ikata amabula yafimuti." \
--gen_text "Cifwile uyu ou bali nankwe eulemulanga ifyakucita." \
--output_dir /project/tts/students/yining_ws/multi_lng/F5-TTS/outputs/iwslt25 \
--load_vocoder_from_local \
--input_filelist /project/tts/students/yining_ws/multi_lng/F5-TTS/inputs/iwslt25/60000_with_info.txt.part4
# --vocoder_local_path /project/tts/students/yining_ws/multi_lng/F5-TTS/ckpts/voco
# --gen_text "Cifwile uyu ou bali nankwe eulemulanga ifyakucita." \