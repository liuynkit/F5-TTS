from __future__ import annotations

import gc
import math
import os
import numpy as np
import torch
import torchaudio
import wandb
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from ema_pytorch import EMA
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from f5_tts.model import CFM
from f5_tts.model.dataset import DynamicBatchSampler, collate_fn
from f5_tts.model.utils import default, exists, seed_everything

# trainer


class Trainer:
    def __init__(
        self,
        model: CFM,
        total_updates,
        learning_rate,
        num_warmup_updates=20000,
        save_per_updates=1000,
        keep_last_n_checkpoints: int = -1,  # -1 to keep all, 0 to not save intermediate, > 0 to keep last N checkpoints
        checkpoint_path=None,
        batch_size=32,
        batch_size_type: str = "sample",
        max_samples=32,
        grad_accumulation_steps=1,
        max_grad_norm=1.0,
        noise_scheduler: str | None = None,
        duration_predictor: torch.nn.Module | None = None,
        logger: str | None = "wandb",  # "wandb" | "tensorboard" | None
        wandb_project="test_e2-tts",
        wandb_run_name="test_run",
        wandb_resume_id: str = None,
        log_samples: bool = False,
        last_per_updates=None,
        accelerate_kwargs: dict = dict(),
        ema_kwargs: dict = dict(),
        bnb_optimizer: bool = False,
        mel_spec_type: str = "vocos",  # "vocos" | "bigvgan"
        is_local_vocoder: bool = False,  # use local path vocoder
        local_vocoder_path: str = "",  # local vocoder path
        resumable_with_seed: int =None
    ):  
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)

        if logger == "wandb" and not wandb.api.api_key:
            logger = None
        self.log_samples = log_samples

        self.accelerator = Accelerator(
            log_with=logger if logger == "wandb" else None,
            kwargs_handlers=[ddp_kwargs],
            gradient_accumulation_steps=grad_accumulation_steps,
            **accelerate_kwargs,
        )

        # Calculate a unique seed for each process
        process_seed = resumable_with_seed + self.accelerator.process_index
        # Set the seed for this process
        seed_everything(process_seed)

        self.logger = logger
        if self.logger == "wandb":
            if exists(wandb_resume_id):
                init_kwargs = {"wandb": {"resume": "allow", "name": wandb_run_name, "id": wandb_resume_id}}
            else:
                init_kwargs = {"wandb": {"resume": "allow", "name": wandb_run_name}}

            self.accelerator.init_trackers(
                project_name=wandb_project,
                init_kwargs=init_kwargs,
                config={
                    "total_updates": total_updates,
                    "learning_rate": learning_rate,
                    "num_warmup_updates": num_warmup_updates,
                    "batch_size": batch_size,
                    "batch_size_type": batch_size_type,
                    "max_samples": max_samples,
                    "grad_accumulation_steps": grad_accumulation_steps,
                    "max_grad_norm": max_grad_norm,
                    "gpus": self.accelerator.num_processes,
                    "noise_scheduler": noise_scheduler,
                },
            )

        elif self.logger == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=f"runs/{wandb_run_name}")

        self.model = model

        if self.is_main:
            self.ema_model = EMA(model, include_online_model=False, **ema_kwargs)
            self.ema_model.to(self.accelerator.device)

            print(f"Using logger: {logger}")
            if grad_accumulation_steps > 1:
                print(
                    "Gradient accumulation checkpointing with per_updates now, old logic per_steps used with before f992c4e"
                )

        self.total_updates = total_updates
        self.num_warmup_updates = num_warmup_updates
        self.save_per_updates = save_per_updates
        self.keep_last_n_checkpoints = keep_last_n_checkpoints
        self.last_per_updates = default(last_per_updates, save_per_updates)
        self.checkpoint_path = default(checkpoint_path, "ckpts/test_e2-tts")

        self.batch_size = batch_size
        self.batch_size_type = batch_size_type
        self.max_samples = max_samples
        self.grad_accumulation_steps = grad_accumulation_steps
        self.max_grad_norm = max_grad_norm

        # mel vocoder config
        self.vocoder_name = mel_spec_type
        self.is_local_vocoder = is_local_vocoder
        self.local_vocoder_path = local_vocoder_path

        self.noise_scheduler = noise_scheduler

        self.duration_predictor = duration_predictor

        if bnb_optimizer:
            import bitsandbytes as bnb

            self.optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=learning_rate)
        else:
            self.optimizer = AdamW(model.parameters(), lr=learning_rate)
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def save_checkpoint(self, update, last=False):
        self.accelerator.wait_for_everyone()
        if self.is_main:
            checkpoint = dict(
                model_state_dict=self.accelerator.unwrap_model(self.model).state_dict(),
                optimizer_state_dict=self.accelerator.unwrap_model(self.optimizer).state_dict(),
                ema_model_state_dict=self.ema_model.state_dict(),
                scheduler_state_dict=self.scheduler.state_dict(),
                update=update,
            )
            if not os.path.exists(self.checkpoint_path):
                os.makedirs(self.checkpoint_path)
            if last:
                self.accelerator.save(checkpoint, f"{self.checkpoint_path}/model_last.pt")
                print(f"Saved last checkpoint at update {update}")
            else:
                if self.keep_last_n_checkpoints == 0:
                    return
                self.accelerator.save(checkpoint, f"{self.checkpoint_path}/model_{update}.pt")
                if self.keep_last_n_checkpoints > 0:
                    # Updated logic to exclude pretrained model from rotation
                    checkpoints = [
                        f
                        for f in os.listdir(self.checkpoint_path)
                        if f.startswith("model_")
                        and not f.startswith("pretrained_")  # Exclude pretrained models
                        and f.endswith(".pt")
                        and f != "model_last.pt"
                    ]
                    checkpoints.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
                    while len(checkpoints) > self.keep_last_n_checkpoints:
                        oldest_checkpoint = checkpoints.pop(0)
                        os.remove(os.path.join(self.checkpoint_path, oldest_checkpoint))
                        print(f"Removed old checkpoint: {oldest_checkpoint}")

    def load_checkpoint(self):
        if (
            not exists(self.checkpoint_path)
            or not os.path.exists(self.checkpoint_path)
            or not any(filename.endswith(".pt") for filename in os.listdir(self.checkpoint_path))
        ):
            return 0

        self.accelerator.wait_for_everyone()
        if "model_last.pt" in os.listdir(self.checkpoint_path):
            latest_checkpoint = "model_last.pt"
        else:
            # Updated to consider pretrained models for loading but prioritize training checkpoints
            all_checkpoints = [
                f
                for f in os.listdir(self.checkpoint_path)
                if (f.startswith("model_") or f.startswith("pretrained_")) and f.endswith(".pt")
            ]

            # First try to find regular training checkpoints
            training_checkpoints = [f for f in all_checkpoints if f.startswith("model_") and f != "model_last.pt"]
            if training_checkpoints:
                latest_checkpoint = sorted(
                    training_checkpoints,
                    key=lambda x: int("".join(filter(str.isdigit, x))),
                )[-1]
            else:
                # If no training checkpoints, use pretrained model
                latest_checkpoint = next(f for f in all_checkpoints if f.startswith("pretrained_"))

        # checkpoint = torch.load(f"{self.checkpoint_path}/{latest_checkpoint}", map_location=self.accelerator.device)  # rather use accelerator.load_state ಥ_ಥ
        checkpoint = torch.load(f"{self.checkpoint_path}/{latest_checkpoint}", weights_only=True, map_location="cpu")

        # patch for backward compatibility, 305e3ea
        for key in ["ema_model.mel_spec.mel_stft.mel_scale.fb", "ema_model.mel_spec.mel_stft.spectrogram.window"]:
            if key in checkpoint["ema_model_state_dict"]:
                del checkpoint["ema_model_state_dict"][key]

        if self.is_main:
            self.ema_model.load_state_dict(checkpoint["ema_model_state_dict"])

        if "update" in checkpoint or "step" in checkpoint:
            # patch for backward compatibility, with before f992c4e
            if "step" in checkpoint:
                checkpoint["update"] = checkpoint["step"] // self.grad_accumulation_steps
                if self.grad_accumulation_steps > 1 and self.is_main:
                    print(
                        "F5-TTS WARNING: Loading checkpoint saved with per_steps logic (before f992c4e), will convert to per_updates according to grad_accumulation_steps setting, may have unexpected behaviour."
                    )
            # patch for backward compatibility, 305e3ea
            for key in ["mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"]:
                if key in checkpoint["model_state_dict"]:
                    del checkpoint["model_state_dict"][key]

            self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint["model_state_dict"])
            self.accelerator.unwrap_model(self.optimizer).load_state_dict(checkpoint["optimizer_state_dict"])
            if self.scheduler:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            update = checkpoint["update"]
        else:
            checkpoint["model_state_dict"] = {
                k.replace("ema_model.", ""): v
                for k, v in checkpoint["ema_model_state_dict"].items()
                if k not in ["initted", "update", "step"]
            }
            self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint["model_state_dict"])
            update = 0

        del checkpoint
        gc.collect()
        return update

    def shuffle_indices_withbuckets(self, train_dataset, random_seed, num_buckets=5):
        # Step 1: Sort dataset into buckets based on sequence length
        # Step 2: Shuffle within each bucket
        # Step 3: Flatten into a final shuffled list
        # Step 4: Use `DistributedSampler` (shuffle=False) on this list
        # Define number of buckets (e.g., 5)
        num_buckets = 5  
        # Get frame lengths for all samples
        frame_lengths = np.array([train_dataset.get_frame_len(i) for i in range(len(train_dataset))])

        # Compute length-based percentiles
        percentiles = np.percentile(frame_lengths, np.linspace(0, 100, num_buckets + 1))

        # Create buckets
        bucketed_indices = [[] for _ in range(num_buckets)]
        for i, length in enumerate(frame_lengths):
            for j in range(num_buckets):
                if percentiles[j] <= length < percentiles[j + 1]:
                    bucketed_indices[j].append(i)
                    break

        np.random.seed(random_seed)  # Ensures different shuffling each epoch
        # Shuffle within each bucket
        for bucket in bucketed_indices:
            np.random.shuffle(bucket)

        # Flatten back into a single shuffled list
        shuffled_indices = [idx for bucket in bucketed_indices for idx in bucket]
        return shuffled_indices

    def validate(self, valid_dataloader: DataLoader, num_workers=16, resumable_with_seed: int = None):
        self.model.eval()  # Set the model to evaluation mode
        total_loss = 0.0
        total_batches = 0

        with torch.no_grad():  # Disable gradient calculation
            for batch in valid_dataloader:
                text_inputs = batch["text"]
                mel_spec = batch["mel"].permute(0, 2, 1)
                mel_lengths = batch["mel_lengths"]

                # Forward pass
                loss, _, _ = self.model(mel_spec, text=text_inputs, lens=mel_lengths, noise_scheduler=self.noise_scheduler)
                total_loss += loss.item()
                total_batches += 1
        
        avg_loss = total_loss / total_batches  # Compute average loss
        self.model.train()
        return avg_loss

    def train(self, train_dataset: Dataset, valid_dataset: Dataset=None, num_workers=16, resumable_with_seed: int = None):
        resumable_with_seed = resumable_with_seed + self.accelerator.process_index

        if self.log_samples:
            from f5_tts.infer.utils_infer import cfg_strength, load_vocoder, nfe_step, sway_sampling_coef

            vocoder = load_vocoder(
                vocoder_name=self.vocoder_name, is_local=self.is_local_vocoder, local_path=self.local_vocoder_path
            )
            target_sample_rate = self.accelerator.unwrap_model(self.model).mel_spec.target_sample_rate
            log_samples_path = f"{self.checkpoint_path}/samples"
            os.makedirs(log_samples_path, exist_ok=True)

        if exists(resumable_with_seed):
            generator = torch.Generator()
            generator.manual_seed(resumable_with_seed)
        else:
            generator = None

        if self.batch_size_type == "sample":
            train_dataloader = DataLoader(
                train_dataset,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=True,
                batch_size=self.batch_size,
                shuffle=True,
                generator=generator,
            )
        elif self.batch_size_type == "frame":
            self.accelerator.even_batches = False
            # sampler = SequentialSampler(train_dataset) # The batch_size and drop_last arguments essentially are used to construct a batch_sampler from sampler.
            shuffled_indices = self.shuffle_indices_withbuckets(train_dataset, random_seed=resumable_with_seed)
            
            # use distributed sampler instead for balanced working load across all gpus
            sampler = DistributedSampler(
                shuffled_indices,  # Now each GPU gets balanced data
                num_replicas=self.accelerator.num_processes,
                rank=self.accelerator.process_index,
                shuffle=False,  # We already shuffled within buckets
            )
            
            batch_sampler = DynamicBatchSampler(
                sampler,
                self.batch_size,
                max_samples=self.max_samples,
                random_seed=resumable_with_seed,  # This enables reproducible shuffling
                drop_last=True,
                data_source=train_dataset
            )
            train_dataloader = DataLoader(
                train_dataset,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=True, # Enable pinned memory for faster GPU transfers
                persistent_workers=True, # Keep workers alive across epochs without reinitialising
                batch_sampler=batch_sampler, # When a batch_sampler is provided, it overrides any shuffling that would normally be done by the DataLoader.
            )
        else:
            raise ValueError(f"batch_size_type must be either 'sample' or 'frame', but received {self.batch_size_type}")

        # Create the validation dataloader
        if valid_dataset:
            self.accelerator.even_batches = False
            sampler = SequentialSampler(valid_dataset) # The batch_size and drop_last arguments essentially are used to construct a batch_sampler from sampler.
            valid_batch_sampler = DynamicBatchSampler(
                sampler,
                self.batch_size,
                max_samples=self.max_samples,
                random_seed=resumable_with_seed,  # This enables reproducible shuffling
                drop_last=True,
            )
            valid_dataloader = DataLoader(
                valid_dataset,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=True, # Enable pinned memory for faster GPU transfers
                persistent_workers=True, # Keep workers alive across epochs without reinitialising
                batch_sampler=valid_batch_sampler, # When a batch_sampler is provided, it overrides any shuffling that would normally be done by the DataLoader.
            )
            valid_dataloader = self.accelerator.prepare(valid_dataloader)
        #  accelerator.prepare() dispatches batches to devices;
        #  which means the length of dataloader calculated before, should consider the number of devices
        # warmup_updates = (
        #     self.num_warmup_updates * self.accelerator.num_processes
        # )  # consider a fixed warmup steps while using accelerate multi-gpu ddp
        warmup_updates = self.num_warmup_updates
        # otherwise by default with split_batches=False, warmup steps change with num_processes
        # total_updates = math.ceil(len(train_dataloader) / self.grad_accumulation_steps) * self.epochs

        epochs = math.ceil(self.total_updates*self.grad_accumulation_steps/len(train_dataloader))
        decay_updates = self.total_updates - warmup_updates
        warmup_scheduler = LinearLR(self.optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_updates)
        decay_scheduler = LinearLR(self.optimizer, start_factor=1.0, end_factor=1e-3, total_iters=decay_updates)
        self.scheduler = SequentialLR(
            self.optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=[warmup_updates]
        )
        train_dataloader, self.scheduler = self.accelerator.prepare(
            train_dataloader, self.scheduler
        )  # actual multi_gpu updates = single_gpu updates / gpu nums
        start_update = self.load_checkpoint()
        global_update = start_update

        if exists(resumable_with_seed):
            orig_epoch_step = len(train_dataloader)
            start_step = start_update * self.grad_accumulation_steps
            # how many full epochs have been finished before resuming
            skipped_epoch = int(start_step // orig_epoch_step)
            # how many batch in the not full epochs have been finished before resuming
            skipped_batch = start_step % orig_epoch_step
            skipped_dataloader = self.accelerator.skip_first_batches(train_dataloader, num_batches=skipped_batch)
        else:
            skipped_epoch = 0

        for epoch in range(skipped_epoch, epochs):
            self.model.train()
            self.accelerator.wait_for_everyone()
            # Synchronize before starting the progress bar
            # torch.distributed.barrier()
            if exists(resumable_with_seed) and epoch == skipped_epoch:
                progress_bar_initial = math.ceil(skipped_batch / self.grad_accumulation_steps)
                current_dataloader = skipped_dataloader
            else:
                progress_bar_initial = 0
                current_dataloader = train_dataloader

            # Set epoch for the batch sampler if it exists
            if hasattr(train_dataloader, "batch_sampler") and hasattr(train_dataloader.batch_sampler, "set_epoch"):
                train_dataloader.batch_sampler.set_epoch(epoch)

            progress_bar = tqdm(
                range(math.ceil(len(train_dataloader) / self.grad_accumulation_steps)),
                desc=f"Epoch {epoch+1}/{epochs}",
                unit="update",
                disable=not self.accelerator.is_local_main_process,
                initial=progress_bar_initial,
            )

            for idx, batch in enumerate(current_dataloader):
                # print("processing batch idx: ", idx)
                # print("batch first 5: ", batch["text"][:5])
                # Handles gradient accumulation automatically.
                # Ensures gradients are accumulated across multiple GPUs before updating weights.
                with self.accelerator.accumulate(self.model):
                    text_inputs = batch["text"]
                    mel_spec = batch["mel"].permute(0, 2, 1)
                    mel_lengths = batch["mel_lengths"]

                    # TODO. add duration predictor training
                    if self.duration_predictor is not None and self.accelerator.is_local_main_process:
                        dur_loss = self.duration_predictor(mel_spec, lens=batch.get("durations"))
                        self.accelerator.log({"duration loss": dur_loss.item()}, step=global_update)

                    # Forward & Backward Pass
                    loss, cond, pred = self.model(
                        mel_spec, text=text_inputs, lens=mel_lengths, noise_scheduler=self.noise_scheduler
                    )
                    self.accelerator.backward(loss)

                    if self.max_grad_norm > 0 and self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                # EMA helps with stable training by maintaining a smoothed version of the model weights.
                # Only the main process updates EMA.
                if self.accelerator.sync_gradients:
                    if self.is_main:
                        self.ema_model.update()

                    global_update += 1
                    progress_bar.update(1)
                    progress_bar.set_postfix(update=str(global_update), loss=loss.item())

                if self.accelerator.is_local_main_process:
                    self.accelerator.log(
                        {"loss": loss.item(), "lr": self.scheduler.get_last_lr()[0]}, step=global_update
                    )
                    if self.logger == "tensorboard":
                        self.writer.add_scalar("loss", loss.item(), global_update)
                        self.writer.add_scalar("lr", self.scheduler.get_last_lr()[0], global_update)

                if global_update % self.save_per_updates == 0 and self.accelerator.sync_gradients:
                    self.save_checkpoint(global_update)

                if global_update % self.last_per_updates == 0 and self.accelerator.sync_gradients:
                    # self.save_checkpoint(global_update, last=True)
                    # add validation loss
                    if valid_dataset:
                        val_loss = self.validate(valid_dataloader, num_workers, resumable_with_seed)
                        if self.accelerator.is_local_main_process:
                            print(f"Validation Loss at update {global_update}: {val_loss:.4f}")
                            self.accelerator.log({"val_loss": val_loss}, step=global_update)

                    if self.log_samples and self.accelerator.is_local_main_process:
                        ref_audio_len = mel_lengths[0]
                        infer_text = [
                            text_inputs[0] + ([" "] if isinstance(text_inputs[0], list) else " ") + text_inputs[0]
                        ]
                        with torch.inference_mode():
                            generated, _ = self.accelerator.unwrap_model(self.model).sample(
                                cond=mel_spec[0][:ref_audio_len].unsqueeze(0),
                                text=infer_text,
                                duration=ref_audio_len * 2,
                                steps=nfe_step,
                                cfg_strength=cfg_strength,
                                sway_sampling_coef=sway_sampling_coef,
                            )
                            generated = generated.to(torch.float32)
                            gen_mel_spec = generated[:, ref_audio_len:, :].permute(0, 2, 1).to(self.accelerator.device)
                            ref_mel_spec = batch["mel"][0].unsqueeze(0)
                            if self.vocoder_name == "vocos":
                                gen_audio = vocoder.decode(gen_mel_spec).cpu()
                                ref_audio = vocoder.decode(ref_mel_spec).cpu()
                            elif self.vocoder_name == "bigvgan":
                                gen_audio = vocoder(gen_mel_spec).squeeze(0).cpu()
                                ref_audio = vocoder(ref_mel_spec).squeeze(0).cpu()
                        torchaudio.save(
                            f"{log_samples_path}/update_{global_update}_gen.wav", gen_audio, target_sample_rate
                        )
                        torchaudio.save(
                            f"{log_samples_path}/update_{global_update}_ref.wav", ref_audio, target_sample_rate
                        )

        self.save_checkpoint(global_update, last=True)

        self.accelerator.end_training()