export PATH="/home/yliu/miniconda3/envs/e2tts/bin:$PATH"
MY_PATH="/project/tts/students/yining_ws/multi_lng/F5-TTS/src"
export CUDA_VISIBLE_DEVICES=1
cd $MY_PATH
# "Setzte man das Verfahren fort, so kamen jedesmal Einfälle," \
# --gen_text "我真的很好奇，这个large language model模型的性能究竟如何。害，有的时候我觉着自己每天就是在瞎忙。Ich muß behaupten, es ist manchmal recht nützlich, Vorurteile zu haben." \
# /project/tts/students/yining_ws/multi_lng/F5-TTS/tests/ref_audio/ber_psychoanalyse_03_f000013.wav \
# --ref_audio /project/tts/students/yining_ws/multi_lng/F5-TTS/dataset/impaired/data_impairedSpeech_new/segments/text_1-usr0227_Balto_0031394_0032000.wav \
# --ref_text "Was kann man tun?" \
# "Die UN-Weltnaturkonferenz COP16 in Kolumbien ist ohne eine Einigung zu Finanzierungsfragen zu Ende gegangen. Der Gipfel war bereits in die Verlängerung gegangen, nachdem bei der eigentlich letzten Plenarsitzung gestern Abend (Ortszeit) keine Einigung erzielt werden konnte." \
# /project/tts/students/yining_ws/multi_lng/F5-TTS/tests/ref_audio/ber_psychoanalyse_03_f000013.wav
# Ich kann in den Wolken ein smiling Gesicht erkennen. Es happened alles ziemlich plötzlich. Du bist ein verzweifelter little Kerl. Linkola hat sowohl in seinem Heimatland als auch weltweit beachtliche controversy entfacht.
# --ref_audio  /project/tts/students/yining_ws/multi_lng/F5-TTS/tests/ref_audio/20081231_neujahrsansprache_f000001.wav \
# --ref_text "Liebe Mitbürgerinnen und Mitbürger, der Jahreswechsel ist die Zeit, einmal Wichtiges von Unwichtigem zu trennen." \
python f5_tts/infer/infer_cli.py \
--model E2-TTS \
--ckpt_file /project/tts/students/yining_ws/multi_lng/F5-TTS/ckpts/Emilia_DE/model_384800.pt \
--vocab_file /project/tts/students/yining_ws/multi_lng/F5-TTS/data/Emilia_ZH_EN_pinyin/vocab.txt \
--ref_audio /project/tts/students/yining_ws/multi_lng/F5-TTS/dataset/impaired/data_impairedSpeech_new/segments/text_1-usr0227_Balto_0031394_0032000.wav \
--ref_text "Was kann man tun?" \
--gen_text "Liebe Mitbürgerinnen und Mitbürger, der Jahreswechsel ist die Zeit, einmal Wichtiges von Unwichtigem zu trennen." \
--output_dir /project/tts/students/yining_ws/multi_lng/F5-TTS/outputs/DE \
--speed 3.0
