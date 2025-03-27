import argparse
import codecs
import os
import re
from datetime import datetime
from importlib.resources import files
from pathlib import Path
import csv
import numpy as np
import soundfile as sf
import tomli
from cached_path import cached_path
from omegaconf import OmegaConf
from tqdm import tqdm

import sys
path_to_add = '/project/tts/students/yining_ws/multi_lng/F5-TTS/src'
# Check if the path is already in sys.path
if path_to_add not in sys.path:
    sys.path.append(path_to_add)
    print(f"Added {path_to_add} to sys.path")
else:
    print(f"{path_to_add} is already in sys.path")

from f5_tts.infer.utils_infer import (
    mel_spec_type,
    target_rms,
    cross_fade_duration,
    nfe_step,
    cfg_strength,
    sway_sampling_coef,
    speed,
    fix_duration,
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
)
from f5_tts.model import DiT, UNetT  # noqa: F401. used for config


parser = argparse.ArgumentParser(
    prog="python3 infer-cli.py",
    description="Commandline interface for E2/F5 TTS with Advanced Batch Processing.",
    epilog="Specify options above to override one or more settings from config.",
)
parser.add_argument(
    "-c",
    "--config",
    type=str,
    default=os.path.join(files("f5_tts").joinpath("infer/examples/basic"), "basic.toml"),
    help="The configuration file, default see infer/examples/basic/basic.toml",
)

# Note. Not to provide default value here in order to read default from config file

parser.add_argument(
    "-m",
    "--model",
    type=str,
    help="The model name: F5TTS_v1_Base | F5TTS_Base | E2TTS_Base | etc.",
)
parser.add_argument(
    "-mc",
    "--model_cfg",
    type=str,
    help="The path to F5-TTS model config file .yaml",
)
parser.add_argument(
    "-p",
    "--ckpt_file",
    type=str,
    help="The path to model checkpoint .pt, leave blank to use default",
)
parser.add_argument(
    "-v",
    "--vocab_file",
    type=str,
    help="The path to vocab file .txt, leave blank to use default",
)
parser.add_argument(
    "-r",
    "--ref_audio",
    type=str,
    help="The reference audio file.",
)
parser.add_argument(
    "-s",
    "--ref_text",
    type=str,
    help="The transcript/subtitle for the reference audio",
)
parser.add_argument(
    "-t",
    "--gen_text",
    type=str,
    help="The text to make model synthesize a speech",
)
parser.add_argument(
    "-f",
    "--gen_file",
    type=str,
    help="The file with text to generate, will ignore --gen_text",
)
parser.add_argument(
    "-o",
    "--output_dir",
    type=str,
    help="The path to output folder",
)
parser.add_argument(
    "-w",
    "--output_file",
    type=str,
    help="The name of output file",
)
parser.add_argument(
    "--save_chunk",
    action="store_true",
    help="To save each audio chunks during inference",
)
parser.add_argument(
    "--remove_silence",
    action="store_true",
    help="To remove long silence found in ouput",
)
parser.add_argument(
    "--load_vocoder_from_local",
    action="store_true",
    help="To load vocoder from local dir, default to ../checkpoints/vocos-mel-24khz",
)
parser.add_argument(
    "--vocoder_name",
    type=str,
    choices=["vocos", "bigvgan"],
    help=f"Used vocoder name: vocos | bigvgan, default {mel_spec_type}",
)
parser.add_argument(
    "--target_rms",
    type=float,
    help=f"Target output speech loudness normalization value, default {target_rms}",
)
parser.add_argument(
    "--cross_fade_duration",
    type=float,
    help=f"Duration of cross-fade between audio segments in seconds, default {cross_fade_duration}",
)
parser.add_argument(
    "--nfe_step",
    type=int,
    help=f"The number of function evaluation (denoising steps), default {nfe_step}",
)
parser.add_argument(
    "--cfg_strength",
    type=float,
    help=f"Classifier-free guidance strength, default {cfg_strength}",
)
parser.add_argument(
    "--sway_sampling_coef",
    type=float,
    help=f"Sway Sampling coefficient, default {sway_sampling_coef}",
)
parser.add_argument(
    "--speed",
    type=float,
    help=f"The speed of the generated audio, default {speed}",
)
parser.add_argument(
    "--fix_duration",
    type=float,
    help=f"Fix the total duration (ref and gen audios) in seconds, default {fix_duration}",
)
parser.add_argument(
    "--input_filelist",
    type=str,
)
parser.add_argument(
    "--wav_save_dir",
    type=str,
)
args = parser.parse_args()


def load_translation_data(file_path, sample_size=60000, output_file=None):
    data = []
    # num = 0
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter='|')
        for row in csv_reader:
            if len(row) == 3:
                english, bemba, data_source = row
                if data_source!='nllb' or len(bemba)>=200:
                    continue
                data.append({
                    # 'english': english,
                    'bemba': bemba,
                    # 'source': data_source
                })
                # num+=1

    # Randomly sample sentences
    if len(data) > sample_size:
        sampled_data = random.sample(data, sample_size)
    else:
        sampled_data = data

    # Write sampled sentences to a txt file
    with open(output_file, 'w', encoding='utf-8') as txtfile:
        for sentence in sampled_data:
            txtfile.write(sentence + '\n')

    print(f"Sampled {len(sampled_data)} sentences and wrote them to {output_file}")
    return sampled_data


def read_and_split_file(file_path):
    result = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                # Strip whitespace and split by '|'
                split_line = [item.strip() for item in line.strip().split('|')]
                if split_line:  # Only add non-empty lines
                    result.append(split_line)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except IOError:
        print(f"Error: Unable to read file '{file_path}'.")
    return result

# inference process
class E2ttsSynthesiser():
    def __init__(self, **kwargs):
        # Handle other keyword arguments if needed
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        for key, value in kwargs.items():
            print('attribute: ', key, getattr(self, key))

        model_cfg = OmegaConf.load(
            self.model_cfg or config.get("model_cfg", str(files("f5_tts").joinpath(f"configs/{self.model}.yaml")))
        )
        config = "/project/tts/students/yining_ws/multi_lng/F5-TTS/src/f5_tts/infer/examples/basic/basic.toml"
        self.config = tomli.load(open(config, "rb"))
        # print('DEBUG: ', self.config)
        self.vocoder = load_vocoder(vocoder_name="vocos", is_local=True, local_path="/project/tts/students/yining_ws/multi_lng/F5-TTS/ckpts/voco")
        self.ema_model = load_model(UNetT, model_cfg.model, self.ckpt_file, mel_spec_type=self.vocoder_name, vocab_file=self.vocab_file)

    def infer_one(self, ref_audio, ref_text, gen_text, idx, output_dir, gen_audio_path=None):
        main_voice = {"ref_audio": ref_audio, "ref_text": ref_text}
        voices = {"main": main_voice}

        for voice in voices:
            print("Voice:", voice)
            print("ref_audio ", voices[voice]["ref_audio"])
            # 
            voices[voice]["ref_audio"], voices[voice]["ref_text"] = preprocess_ref_audio_text(
                voices[voice]["ref_audio"], voices[voice]["ref_text"], clip_short=False
            )
            print("ref_text_", voices[voice]["ref_text"], "\n\n")
            print("ref_audio_", voices[voice]["ref_audio"], "\n\n")

        # "[announcer]Welcome! [main]This is a test."
        generated_audio_segments = []
        reg1 = r"(?=\[\w+\])"
        chunks = re.split(reg1, gen_text)
        reg2 = r"\[(\w+)\]"
        for text in chunks:
            if not text.strip():
                continue
            match = re.match(reg2, text)
            # It checks if the specified voice exists in the voices dictionary, falling back to "main" if not found.
            if match:
                voice = match[1]
            else:
                print("No voice tag found, using main.")
                voice = "main"
            if voice not in voices:
                print(f"Voice {voice} not found, using main.")
                voice = "main"
            text = re.sub(reg2, "", text)
            ref_audio_ = voices[voice]["ref_audio"]
            ref_text_ = voices[voice]["ref_text"]
            gen_text_ = text.strip()
            print(f"Voice: {voice}")
            # TODO check how the sampling rate is setted
            audio_segment, final_sample_rate, spectragram = infer_process(
                ref_audio_,
                ref_text_,
                gen_text_,
                self.ema_model,
                self.vocoder,
                mel_spec_type=self.vocoder_name or self.config.get("vocoder_name", mel_spec_type),
                target_rms=self.target_rms or self.config.get("target_rms", target_rms),
                cross_fade_duration=self.cross_fade_duration or self.config.get("cross_fade_duration", cross_fade_duration),
                nfe_step=self.nfe_step or self.config.get("nfe_step", nfe_step),
                cfg_strength=self.cfg_strength or self.config.get("cfg_strength", cfg_strength),
                sway_sampling_coef=self.sway_sampling_coef or self.config.get("sway_sampling_coef", sway_sampling_coef),
                speed=self.speed or self.config.get("speed", speed),
                fix_duration=self.fix_duration or self.config.get("fix_duration", fix_duration),
            )
            generated_audio_segments.append(audio_segment)

        if generated_audio_segments:
            final_wave = np.concatenate(generated_audio_segments)

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            if gen_audio_path:
                wave_path = output_dir+"/"+gen_audio_path
            else:
                wave_path = output_dir+"/augmented_"+str(idx)+".wav"
            with open(wave_path, "wb") as f:
                sf.write(f.name, final_wave, final_sample_rate)
                # Remove silence
                remove_silence = self.remove_silence or self.config.get("remove_silence", False)
                if remove_silence:
                    remove_silence_for_generated_wav(f.name)
                print(f.name)

        return "augmented_"+str(idx)+".wav"
    
    def infer_all(self, file_path, wav_save_dir, output_txt_file):

        print("file_path: ", file_path)

        progress_bar = tqdm(
                range(10),
                desc=f"Audio Files",
                unit="update",
                initial=0,
        )
        start_index = 0
        # start_index = (int(file_path[-1])-1)*15000+1
        # print("start_index: ", start_index)
        # 1----1   2----15001  3-30001 4-45001

        results = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for idx, line in enumerate(file):
                
                # Strip whitespace and split by '|'
                split_line = [item.strip().strip('\n') for item in line.strip().split('|')]
                if len(split_line) == 3:  # Only add non-empty lines
                    gen_text, ref_audio, ref_text = split_line
                    items = []

                    gen_audio = self.infer_one(ref_audio, ref_text, gen_text, idx+start_index, wav_save_dir)
                    items.append(gen_audio)
                    items +=split_line

                    results.append("|".join(items))
                    progress_bar.update(1)
                    # progress_bar.set_postfix(update=str(global_update), loss=loss.item())
        # Write sampled sentences to a txt file
        with open(output_txt_file, 'w', encoding='utf-8') as txtfile:
            for result in results:
                txtfile.write(result + '\n')

    def infer_all_following_idx(self, file_path, wav_save_dir, output_txt_file):

        print("file_path: ", file_path)

        progress_bar = tqdm(
                range(10),
                desc=f"Audio Files",
                unit="update",
                initial=0,
        )
        start_index = 0
        # start_index = (int(file_path[-1])-1)*15000+1
        # print("start_index: ", start_index)
        # 1----1   2----15001  3-30001 4-45001

        # results = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for idx, line in enumerate(file):
                # Strip whitespace and split by '|'
                split_line = [item.strip().strip('\n') for item in line.strip().split('|')]
                if len(split_line) == 4:  # Only add non-empty lines
                    gen_audio, gen_text, ref_audio, ref_text = split_line
                    items = []

                    _ = self.infer_one(ref_audio, ref_text, gen_text, idx+start_index, wav_save_dir, gen_audio_path=gen_audio)
                    # items.append(gen_audio)
                    # items +=split_line
                    # results.append("|".join(items))
                    progress_bar.update(1)
                    # progress_bar.set_postfix(update=str(global_update), loss=loss.item()


def main():
    synthesiser = E2ttsSynthesiser(**vars(args))
    ref_audio = "/project/OML/zli/iwslt2024/low_resourced_track/data/bem_eng/training/bigc/bigc/data/bem/audio/26_8819_0_261_01_211112-010515_bem_8f1_elicit_0.wav"
    ref_text = "Abalumendo babili nabeminina bale ikata amabula yafimuti."
    gen_text = "Cifwile uyu ou bali nankwe eulemulanga ifyakucita."
    # wav_save_dir = "/project/tts/students/yining_ws/multi_lng/F5-TTS/outputs/iwslt25_augmented_60000/wavs"
    # synthesiser.infer_one(ref_audio, ref_text, gen_text, 0, wav_save_dir)
    # txt_file = "/project/tts/students/yining_ws/multi_lng/F5-TTS/inputs/iwslt25/10_with_info.txt"
    output_txt_file = "/project/tts/students/yining_ws/multi_lng/F5-TTS/outputs/iwslt25_augmented_test/filelist.txt"
    synthesiser.infer_all_following_idx(args.input_filelist, args.wav_save_dir, output_txt_file)


if __name__ == "__main__":
    main()