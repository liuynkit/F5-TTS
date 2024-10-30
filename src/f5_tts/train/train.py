# training script.

import os
from importlib.resources import files

import hydra
import cProfile
import pstats

import sys
import torch
import time

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

path_to_add = '/project/tts/students/yining_ws/multi_lng/F5-TTS/src'

# Check if the path is already in sys.path
if path_to_add not in sys.path:
    sys.path.append(path_to_add)
    print(f"Added {path_to_add} to sys.path")
else:
    print(f"{path_to_add} is already in sys.path")

from f5_tts.model import CFM, DiT, Trainer, UNetT
from f5_tts.model.dataset import load_dataset
from f5_tts.model.utils import get_tokenizer

os.chdir(str(files("f5_tts").joinpath("../..")))  # change working directory to root of project (local editable)
# os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
os.environ['NCCL_DEBUG'] = 'INFO'
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def init_accelerator():
    from accelerate import Accelerator
    return Accelerator()


def warmup_gpu(device="cuda:0"):
    """
    Warm up the GPU by performing simple operations to initialize the CUDA context.
    """
    # Specify the device
    device = torch.device(device)

    # Allocate a tensor on the GPU
    x = torch.randn(1024, 1024, device=device)

    # Perform a simple computation
    y = torch.matmul(x, x)

    # Synchronize to ensure all operations are completed
    torch.cuda.synchronize()



@hydra.main(version_base="1.3", config_path=str(files("f5_tts").joinpath("configs/iwslt25")), config_name=None)
def main(cfg):
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    print("Current CUDA device:", torch.cuda.current_device())
    print("CUDA device name:", torch.cuda.get_device_name(0)) 

    warmup_gpu(device="cuda:0")
    warmup_gpu(device="cuda:1")
    print("config_path: ", str(files("f5_tts").joinpath("configs")))
    print("config: ", cfg)
    mel_spec_type = cfg.model.mel_spec.mel_spec_type
    exp_name = f"{cfg.model.name}_{mel_spec_type}_{cfg.model.tokenizer}_{cfg.datasets.name}"

    # Use the tokenizer and tokenizer_path provided in the config
    tokenizer = cfg.model.tokenizer
    if tokenizer == "custom":
        if not cfg.model.tokenizer_path:
            raise ValueError("Custom tokenizer selected, but no tokenizer_path provided.")
        tokenizer_path = cfg.model.tokenizer_path
    else:
        tokenizer_path = cfg.datasets.name

    print("tokenizer: ", tokenizer)
    print("tokenizer_path: ", tokenizer_path)
    vocab_char_map, vocab_size = get_tokenizer(tokenizer_path, tokenizer)

    # set model
    if "F5TTS" in cfg.model.name:
        model_cls = DiT
    elif "E2TTS" in cfg.model.name:
        model_cls = UNetT
    wandb_resume_id = None

    model = CFM(
        transformer=model_cls(**cfg.model.arch, text_num_embeds=vocab_size, mel_dim=cfg.model.mel_spec.n_mel_channels),
        mel_spec_kwargs=cfg.model.mel_spec,
        vocab_char_map=vocab_char_map,
    )

    print("DEBUG")

    # logger = "wandb"
    # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    # accelerate_kwargs={}
    # accelerator = Accelerator(
    #         # log_with=logger if logger == "wandb" else None,
    #         kwargs_handlers=[ddp_kwargs],
    #         gradient_accumulation_steps=cfg.optim.grad_accumulation_steps,
    #         **accelerate_kwargs,
    # 
    accelerator = Accelerator()

    # accelerator = Accelerator(distributed_type="NO")  # Forces single GPU mode

    # cProfile.run('init_accelerator()', 'accelerator_profile')

    # print("WHEn DEBUG")
    # with open('/project/tts/students/yining_ws/multi_lng/F5-TTS/outputs/accelerator_profile_stats.txt', 'w') as f:
    #     p = pstats.Stats('accelerator_profile', stream=f)
    #     p.sort_stats('cumulative').print_stats(30)

    # init trainer
    trainer = Trainer(
        model,
        total_updates=cfg.optim.epochs,
        learning_rate=cfg.optim.learning_rate,
        num_warmup_updates=cfg.optim.num_warmup_updates,
        save_per_updates=cfg.ckpts.save_per_updates,
        keep_last_n_checkpoints=getattr(cfg.ckpts, "keep_last_n_checkpoints", -1),
        checkpoint_path=str(files("f5_tts").joinpath(f"../../{cfg.ckpts.save_dir}")),
        batch_size=cfg.datasets.batch_size_per_gpu,
        batch_size_type=cfg.datasets.batch_size_type,
        max_samples=cfg.datasets.max_samples,
        grad_accumulation_steps=cfg.optim.grad_accumulation_steps,
        max_grad_norm=cfg.optim.max_grad_norm,
        logger=cfg.ckpts.logger,
        wandb_project="CFM-TTS",
        wandb_run_name=exp_name,
        wandb_resume_id=wandb_resume_id,
        last_per_updates=cfg.ckpts.last_per_updates,
        log_samples=True,
        bnb_optimizer=cfg.optim.bnb_optimizer,
        mel_spec_type=mel_spec_type,
        is_local_vocoder=cfg.model.vocoder.is_local,
        local_vocoder_path=cfg.model.vocoder.local_path,
    )
    print('cfg: ', cfg)

    train_dataset = load_dataset(cfg.datasets.name, tokenizer, mel_spec_kwargs=cfg.model.mel_spec)
    trainer.train(
        train_dataset,
        num_workers=cfg.datasets.num_workers,
        resumable_with_seed=666,  # seed for shuffling dataset
    )


if __name__ == "__main__":
    main()