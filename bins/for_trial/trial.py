import fasttext
# https://fasttext.cc/docs/en/language-identification.html
model_file = "/project/tts/students/yining_ws/multi_lng/vits/checkpoints/lang_recognition/lid.176.bin"
# if not model_file.is_file():
#     raise FileNotFoundError('Run wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin in'
#                                                 ' ' + str(model_file.parent))
lang_id_model = fasttext.load_model(str(model_file))

text = "I'm a teacher."
(language,), prob = lang_id_model.predict(text)

language = language[len('__label__'):]
print(language)
print(prob)