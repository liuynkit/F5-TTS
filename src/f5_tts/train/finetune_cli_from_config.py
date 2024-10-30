import os
from importlib.resources import files
import shutil
import hydra

import sys
path_to_add = '/project/tts/students/yining_ws/multi_lng/F5-TTS/src'
# Check if the path is already in sys.path
if path_to_add not in sys.path:
    sys.path.append(path_to_add)
    print(f"Added {path_to_add} to sys.path")
else:
    print(f"{path_to_add} is already in sys.path")

from cached_path import cached_path

from f5_tts.model import CFM, UNetT, DiT, Trainer
from f5_tts.model.utils import get_tokenizer
from f5_tts.model.dataset import load_dataset

# -------------------------- Training Settings -------------------------- #
os.chdir(str(files("f5_tts").joinpath("../..")))  # change working directory to root of project (local editable)
os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"

@hydra.main(version_base="1.3", config_path=str(files("f5_tts").joinpath("configs/iwslt25")), config_name=None)
def main(cfg):
    print("config_path: ", str(files("f5_tts").joinpath("configs")))
    print("config: ", cfg)
    ### set path to save training logs
    checkpoint_path = str(files("f5_tts").joinpath(f"../../{cfg.ckpts.save_dir}"))
    ### set run name for wandb
    exp_name = f"{cfg.model.name}_{cfg.model.mel_spec.mel_spec_type}_{cfg.model.tokenizer}_{cfg.datasets.name}"

    # Model parameters based on experiment name
    if cfg.model.name == "F5TTS_v1_Base":
        wandb_resume_id = None
        model_cls = DiT
        model_cfg = dict(
            dim=1024,
            depth=22,
            heads=16,
            ff_mult=2,
            text_dim=512,
            conv_layers=4,
        )
        if cfg.mode.finetune:
            if cfg.mode.pretrain is None:
                ckpt_path = str(cached_path("hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors"))
            else:
                ckpt_path = cfg.mode.pretrain

    elif cfg.model.name == "F5TTS_Base":
        wandb_resume_id = None
        model_cls = DiT
        model_cfg = dict(
            dim=1024,
            depth=22,
            heads=16,
            ff_mult=2,
            text_dim=512,
            text_mask_padding=False,
            conv_layers=4,
            pe_attn_head=1,
        )
        if cfg.mode.finetune:
            if cfg.mode.pretrain is None:
                ckpt_path = str(cached_path("hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.pt"))
            else:
                ckpt_path = cfg.mode.pretrain

    elif cfg.model.name == "E2TTS_Base":
        wandb_resume_id = None
        model_cls = UNetT
        model_cfg = dict(
            dim=1024,
            depth=24,
            heads=16,
            ff_mult=4,
            text_mask_padding=False,
            pe_attn_head=1,
        )
        if cfg.mode.finetune:
            if cfg.mode.pretrain is None:
                ckpt_path = str(cached_path("hf://SWivid/E2-TTS/E2TTS_Base/model_1200000.pt"))
            else:
                ckpt_path = cfg.mode.pretrain

    if cfg.mode.finetune:
        if not os.path.isdir(checkpoint_path):
            os.makedirs(checkpoint_path, exist_ok=True)

        file_checkpoint = os.path.basename(ckpt_path)
        if not file_checkpoint.startswith("pretrained_"):  # Change: Add 'pretrained_' prefix to copied model
            file_checkpoint = "pretrained_" + file_checkpoint
        file_checkpoint = os.path.join(checkpoint_path, file_checkpoint)
        if not os.path.isfile(file_checkpoint):
            shutil.copy2(ckpt_path, file_checkpoint)
            print("copy checkpoint for finetune")

    # Use the tokenizer and tokenizer_path provided in the config
    ### set text tokenizer
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

    ### init model 
    model = CFM(
        transformer=model_cls(**cfg.model.arch, text_num_embeds=vocab_size, mel_dim=cfg.model.mel_spec.n_mel_channels),
        mel_spec_kwargs=cfg.model.mel_spec,
        vocab_char_map=vocab_char_map,
    )

    # seed for shuffling dataset
    seed = 1995

    trainer = Trainer(
        model,
        total_updates=cfg.optim.total_updates,
        learning_rate=cfg.optim.learning_rate,
        num_warmup_updates=cfg.optim.num_warmup_updates,
        save_per_updates=cfg.ckpts.save_per_updates,
        keep_last_n_checkpoints=getattr(cfg.ckpts, "keep_last_n_checkpoints", -1),
        checkpoint_path=checkpoint_path,
        batch_size=cfg.datasets.batch_size_per_gpu,
        batch_size_type=cfg.datasets.batch_size_type,
        max_samples=cfg.datasets.max_samples,
        grad_accumulation_steps=cfg.optim.grad_accumulation_steps,
        max_grad_norm=cfg.optim.max_grad_norm,
        logger=cfg.ckpts.logger,
        wandb_project=cfg.datasets.name+'_'+cfg.model.tokenizer,
        wandb_run_name=exp_name,
        wandb_resume_id=wandb_resume_id,
        log_samples=True,
        last_per_updates=cfg.ckpts.last_per_updates,
        bnb_optimizer=cfg.optim.bnb_optimizer,
        mel_spec_type=cfg.model.mel_spec.mel_spec_type,
        is_local_vocoder=cfg.model.vocoder.is_local,
        local_vocoder_path=cfg.model.vocoder.local_path,
        resumable_with_seed=seed,
    )

    train_dataset = load_dataset(cfg.datasets.name, tokenizer, mel_spec_kwargs=cfg.model.mel_spec)
    validation_dataset = load_dataset(cfg.datasets.name, tokenizer, mel_spec_kwargs=cfg.model.mel_spec, is_valid=True)

    print("Start training...")
    trainer.train(
        train_dataset,
        valid_dataset=validation_dataset,
        num_workers=cfg.datasets.num_workers,
        resumable_with_seed=seed,
    )


if __name__ == "__main__":
    main()