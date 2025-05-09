import argparse
import codecs
import os
import re
from pathlib import Path
from importlib.resources import files
import torch
import numpy as np
import soundfile as sf
import tomli
from cached_path import cached_path

import sys
import onnx
path_to_add = '/project/tts/students/yining_ws/multi_lng/F5-TTS/src'
# Check if the path is already in sys.path
if path_to_add not in sys.path:
    sys.path.append(path_to_add)
    print(f"Added {path_to_add} to sys.path")
else:
    print(f"{path_to_add} is already in sys.path")

from f5_tts.model import DiT, UNetT
from f5_tts.infer.utils_infer import (
    load_vocoder,
    load_model,
    preprocess_ref_audio_text,
    infer_process,
    remove_silence_for_generated_wav,
)


parser = argparse.ArgumentParser(
    prog="python3 infer-cli.py",
    description="Commandline interface for E2/F5 TTS with Advanced Batch Processing.",
    epilog="Specify options above to override one or more settings from config.",
)
parser.add_argument(
    "-c",
    "--config",
    help="Configuration file. Default=infer/examples/basic/basic.toml",
    default=os.path.join(files("f5_tts").joinpath("infer/examples/basic"), "basic.toml"),
)
parser.add_argument(
    "-m",
    "--model",
    help="F5-TTS | E2-TTS",
)
parser.add_argument(
    "-p",
    "--ckpt_file",
    help="The Checkpoint .pt",
)
parser.add_argument(
    "-v",
    "--vocab_file",
    help="The vocab .txt",
)
parser.add_argument("-r", "--ref_audio", type=str, help="Reference audio file < 15 seconds.")
parser.add_argument("-s", "--ref_text", type=str, default="666", help="Subtitle for the reference audio.")
parser.add_argument(
    "-t",
    "--gen_text",
    type=str,
    help="Text to generate.",
)
parser.add_argument(
    "-f",
    "--gen_file",
    type=str,
    help="File with text to generate. Ignores --text",
)
parser.add_argument(
    "-o",
    "--output_dir",
    type=str,
    help="Path to output folder..",
)
parser.add_argument(
    "--remove_silence",
    help="Remove silence.",
)
parser.add_argument(
    "--load_vocoder_from_local",
    action="store_true",
    help="load vocoder from local. Default: ../checkpoints/charactr/vocos-mel-24khz",
)
parser.add_argument(
    "--speed",
    type=float,
    default=1.0,
    help="Adjust the speed of the audio generation (default: 1.0)",
)
args = parser.parse_args()

config = tomli.load(open(args.config, "rb"))

ref_audio = args.ref_audio if args.ref_audio else config["ref_audio"]
ref_text = args.ref_text if args.ref_text != "666" else config["ref_text"]
gen_text = args.gen_text if args.gen_text else config["gen_text"]
gen_file = args.gen_file if args.gen_file else config["gen_file"]

# patches for pip pkg user
if "infer/examples/" in ref_audio:
    ref_audio = str(files("f5_tts").joinpath(f"{ref_audio}"))
if "infer/examples/" in gen_file:
    gen_file = str(files("f5_tts").joinpath(f"{gen_file}"))
if "voices" in config:
    for voice in config["voices"]:
        voice_ref_audio = config["voices"][voice]["ref_audio"]
        if "infer/examples/" in voice_ref_audio:
            config["voices"][voice]["ref_audio"] = str(files("f5_tts").joinpath(f"{voice_ref_audio}"))

if gen_file:
    gen_text = codecs.open(gen_file, "r", "utf-8").read()
output_dir = args.output_dir if args.output_dir else config["output_dir"]
model = args.model if args.model else config["model"]
ckpt_file = args.ckpt_file if args.ckpt_file else ""
vocab_file = args.vocab_file if args.vocab_file else ""
remove_silence = args.remove_silence if args.remove_silence else config["remove_silence"]
speed = args.speed
wave_path = Path(output_dir) / "infer_cli_out.wav"
# spectrogram_path = Path(output_dir) / "infer_cli_out.png"
vocos_local_path = "../checkpoints/charactr/vocos-mel-24khz"

vocoder = load_vocoder(is_local=args.load_vocoder_from_local, local_path=vocos_local_path)


# load models
if model == "F5-TTS":
    model_cls = DiT
    model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
    if ckpt_file == "":
        repo_name = "F5-TTS"
        exp_name = "F5TTS_Base"
        ckpt_step = 1200000
        ckpt_file = str(cached_path(f"hf://SWivid/{repo_name}/{exp_name}/model_{ckpt_step}.safetensors"))
        # ckpt_file = f"ckpts/{exp_name}/model_{ckpt_step}.pt"  # .pt | .safetensors; local path

elif model == "E2-TTS":
    model_cls = UNetT
    model_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4)
    if ckpt_file == "":
        repo_name = "E2-TTS"
        exp_name = "E2TTS_Base"
        ckpt_step = 1200000
        ckpt_file = str(cached_path(f"hf://SWivid/{repo_name}/{exp_name}/model_{ckpt_step}.safetensors"))
        # ckpt_file = f"ckpts/{exp_name}/model_{ckpt_step}.pt"  # .pt | .safetensors; local path

print(f"Using {model}...")
ema_model = load_model(model_cls, model_cfg, ckpt_file, vocab_file)


def main_process(ref_audio, ref_text, text_gen, model_obj, remove_silence, speed):
    main_voice = {"ref_audio": ref_audio, "ref_text": ref_text}
    if "voices" not in config:
        voices = {"main": main_voice}
    else:
        voices = config["voices"]
        voices["main"] = main_voice
    for voice in voices:
        voices[voice]["ref_audio"], voices[voice]["ref_text"] = preprocess_ref_audio_text(
            voices[voice]["ref_audio"], voices[voice]["ref_text"]
        )
        print("Voice:", voice)
        print("Ref_audio:", voices[voice]["ref_audio"])
        print("Ref_text:", voices[voice]["ref_text"])

    generated_audio_segments = []
    reg1 = r"(?=\[\w+\])"
    chunks = re.split(reg1, text_gen)
    reg2 = r"\[(\w+)\]"
    for text in chunks:
        match = re.match(reg2, text)
        if match:
            voice = match[1]
        else:
            print("No voice tag found, using main.")
            voice = "main"
        if voice not in voices:
            print(f"Voice {voice} not found, using main.")
            voice = "main"
        text = re.sub(reg2, "", text)
        gen_text = text.strip()
        ref_audio = voices[voice]["ref_audio"]
        ref_text = voices[voice]["ref_text"]
        print(f"Voice: {voice}")
        audio, final_sample_rate, spectragram = infer_process(
            ref_audio, ref_text, gen_text, model_obj, vocoder, speed=speed
        )
        generated_audio_segments.append(audio)

    if generated_audio_segments:
        final_wave = np.concatenate(generated_audio_segments)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(wave_path, "wb") as f:
            sf.write(f.name, final_wave, final_sample_rate)
            # Remove silence
            if remove_silence:
                remove_silence_for_generated_wav(f.name)
            print(f.name)


class ModelWrapper(torch.nn.Module):
    def __init__(self, model, vocoder):
        super().__init__()
        model = model.float()
        vocoder = vocoder.float()
        self.model = model
        self.vocoder = vocoder

    def forward(self, cond, text, duration, steps, cfg_strength, sway_sampling_coef, ref_audio_len):
        # inference
        with torch.inference_mode():
            out, _ = self.model.inference_for_onnx(
                cond=cond,
                text=text,
                duration=duration,
                steps=steps,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
                # ref_audio_len=ref_audio_len,
                vocoder=self.vocoder 
            )
            # generated = out.to(torch.float32)
            # # generated = generated[:, ref_audio_len:, :]
            # generated_mel_spec = generated.permute(0, 2, 1)
            # generated_wave = self.vocoder.decode(generated_mel_spec.cpu())
            # if rms < target_rms:
            #     generated_wave = generated_wave * rms / target_rms
            # print('generated_wave size: ', generated_wave.size())
            # wav -> numpy
            # generated_wave = generated_wave.squeeze().cpu().numpy()
        # return generated_wave, generated_mel_spec
        return out

def main():
    print("#"*50)
    print("ref_audio: ", ref_audio)
    print("ref_text: ", ref_text)
    print("gen_text: ", gen_text)
    print("ema_model: ", ema_model)
    print("remove_silence: ", remove_silence)
    print("speed: ", speed)
    print("#"*50)
    # main_process(ref_audio, ref_text, gen_text, ema_model, remove_silence, speed)

    # Assuming model_obj is your pretrained model
    model = ModelWrapper(ema_model, vocoder)
    model.eval()

    # Prepare dummy inputs (adjust shapes and types as necessary)
    # dummy_cond = torch.randn(1, 120000).cuda()  # Assuming 10 second of audio at 24kHz
    
    # print(dummy_cond)
    # print(dummy_cond.dtype)

    # dummy_text = ["你好，世界"]  # Assuming this is how text is input
    # dummy_duration = 863  # Assuming duration is an integer
    dummy_steps = 32
    dummy_cfg_strength = 2.0
    dummy_sway_sampling_coef = -1.0

    dummy_cond = torch.randn(1, 500, 100).cuda()
    # Create the text tensor
    dummy_text = torch.randint(0, 1500, (1, 90)).cuda()
    # Create the duration tensor
    dummy_duration = torch.randint(900, 1000, (1,)).cuda()

    dummy_ref_audio_len = torch.randint(0, 1500, (1,)).cuda()
    print('dummy_ref_audio_len.size: ', dummy_ref_audio_len.size())
    # Export the model
    torch.onnx.export(model,
        (dummy_cond, dummy_text, dummy_duration, dummy_steps, dummy_cfg_strength, dummy_sway_sampling_coef, dummy_ref_audio_len),
        "/project/tts/students/yining_ws/multi_lng/F5-TTS/outputs/DE_model.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['cond', 'text', 'duration', 'steps', 'cfg_strength', 'sway_sampling_coef', 'ref_audio_len'],
        output_names=['output'],
        dynamic_axes={'cond': {0: 'batch_size', 1: 'audio_length'}, 
            'text': {0: 'batch_size', 1: 'seq_length'},
            'output': {0: 'batch_size', 1: 'audio_length'},
            'duration': {0: 'batch_size'},
            'ref_audio_len': {0: 'batch_size'}
        }
    ) 

if __name__ == "__main__":
    main()
