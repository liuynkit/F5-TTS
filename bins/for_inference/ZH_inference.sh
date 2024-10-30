export PATH="/home/yliu/miniconda3/envs/e2tts/bin:$PATH"
MY_PATH="/project/tts/students/yining_ws/multi_lng/F5-TTS/src"
export CUDA_VISIBLE_DEVICES=0
cd $MY_PATH
# --gen_text "我真的很好奇，这个large language model模型的性能究竟如何。害，有的时候我觉着自己每天就是在瞎忙。" \
python f5_tts/infer/infer_cli.py \
--model E2-TTS \
--ckpt_file /project/tts/students/yining_ws/multi_lng/F5-TTS/ckpts/Emilia_ZH_EN/model_1200000.pt \
--vocab_file /project/tts/students/yining_ws/multi_lng/F5-TTS/data/Emilia_ZH_EN_pinyin/vocab.txt \
--ref_audio /project/tts/students/yining_ws/multi_lng/F5-TTS/src/f5_tts/infer/examples/basic/basic_ref_zh.wav \
--ref_text "对，这就是我。万人景仰的太乙真人。" \
--gen_text "差不多过去十年了，华裔物理学家郗小星依然清楚记得2015年5月的一天发生的事。那是早上7点左右，他和太太以及两个女儿还在睡觉，突然听到急促的敲门。 “敲的非常用力，声音非常大，是一种你从来没听过的敲门声，感觉门就要应声倒下了”。匆忙中，他只穿上短裤就去开门。他看到约十几人站在门口，有些人带着枪。这些人自称是联邦调查局人员，在确认郗小星的身分后，他们用手铐将他铐起来。紧接着，几个人冲进来房子，大声叫喊发出命令。郗小星的太太从卧室走出，举着枪的调查员命令她举起手走出来。随后他的两个女儿也走出房间。美国联邦调查局以涉嫌经济间谍活动逮捕了这位在中国出生的超导技术领域的知名学者，指控他犯下四项电信诈欺罪，涉及帮助中国发展超导领域的竞争力。" \
--output_dir /project/tts/students/yining_ws/multi_lng/F5-TTS/outputs/ZH
