MY_PATH="/project/tts/students/yining_ws/multi_lng/F5-TTS/src"
export CUDA_VISIBLE_DEVICES=4
cd $MY_PATH
python f5_tts/infer/infer_cli_from_dataset.py \
--model E2-TTS \
--model_cfg /project/tts/students/yining_ws/multi_lng/F5-TTS/src/f5_tts/configs/iwslt25/E2TTS_Base_train_bigc_ben.yaml \
--ckpt_file /project/tts/students/yining_ws/multi_lng/F5-TTS/ckpts/iwslt25/bigc_ben/E2TTS_Base_vocos_char_bigc_bem/checkpoints/model_last.pt \
--vocab_file /project/tts/students/yining_ws/multi_lng/F5-TTS/preprocessed_data/bigc_bem_char/vocab.txt \
--ref_audio /project/OML/zli/iwslt2024/low_resourced_track/data/bem_eng/training/bigc/bigc/data/bem/audio/26_8819_0_261_01_211112-010515_bem_8f1_elicit_0.wav \
--ref_text "Abalumendo babili nabeminina bale ikata amabula yafimuti." \
--gen_text "Abalumendo babili nabeminina bale ikata amabula yafimuti." \
--output_dir /project/tts/students/yining_ws/multi_lng/F5-TTS/outputs/iwslt25
# --load_vocoder_from_local
# --gen_text "Cifwile uyu ou bali nankwe eulemulanga ifyakucita." \