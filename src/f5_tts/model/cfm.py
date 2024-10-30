"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

from __future__ import annotations
from typing import Callable
from random import random

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from torchdiffeq import odeint

from f5_tts.model.modules import MelSpec
from f5_tts.model.utils import (
    default,
    exists,
    list_str_to_idx,
    list_str_to_tensor,
    lens_to_mask,
    mask_from_frac_lengths,
)

# custion pad_sequence for onnx export
def custom_pad_sequence(sequences, padding_value=0, batch_first=True):
    max_len = max([s.size(0) for s in sequences])
    out_tensors = []
    for tensor in sequences:
        padding_size = max_len - tensor.size(0)
        padded = torch.nn.functional.pad(tensor, (0, 0, 0, padding_size), value=padding_value)
        out_tensors.append(padded)
    return torch.stack(out_tensors, dim=0)


class CFM(nn.Module):
    def __init__(
        self,
        transformer: nn.Module,
        sigma=0.0,
        odeint_kwargs: dict = dict(
            # atol = 1e-5,
            # rtol = 1e-5,
            method="euler"  # 'midpoint'
        ),
        audio_drop_prob=0.3,
        cond_drop_prob=0.2,
        num_channels=None,
        mel_spec_module: nn.Module | None = None,
        mel_spec_kwargs: dict = dict(),
        frac_lengths_mask: tuple[float, float] = (0.7, 1.0),
        vocab_char_map: dict[str:int] | None = None,
    ):
        super().__init__()

        self.frac_lengths_mask = frac_lengths_mask

        # mel spec
        self.mel_spec_kwargs = mel_spec_kwargs
        self.mel_spec = default(mel_spec_module, MelSpec(**mel_spec_kwargs))
        num_channels = default(num_channels, self.mel_spec.n_mel_channels)
        self.num_channels = num_channels

        # classifier-free guidance
        self.audio_drop_prob = audio_drop_prob
        self.cond_drop_prob = cond_drop_prob

        # transformer
        self.transformer = transformer
        dim = transformer.dim
        self.dim = dim

        # conditional flow related
        self.sigma = sigma

        # sampling related
        self.odeint_kwargs = odeint_kwargs

        # vocab map for tokenization
        self.vocab_char_map = vocab_char_map

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def sample(
        self,
        cond: float["b n d"] | float["b nw"],  # noqa: F722
        text: int["b nt"] | list[str],  # noqa: F722
        duration: int | int["b"],  # noqa: F821
        *,
        lens: int["b"] | None = None,  # noqa: F821
        steps=32,
        cfg_strength=1.0,
        sway_sampling_coef=None,
        seed: int | None = None,
        max_duration=4096,
        vocoder: Callable[[float["b d n"]], float["b nw"]]
        | None = None,  # noqa: F722
        no_ref_audio=False,
        duplicate_test=False,
        t_inter=0.1,
        edit_mask=None,
    ):
        self.eval()

        if next(self.parameters()).dtype == torch.float16:
            cond = cond.half()

        # raw wave

        if cond.ndim == 2:
            cond = self.mel_spec(cond)

            cond = cond.permute(0, 2, 1)
            assert cond.shape[-1] == self.num_channels

        batch, cond_seq_len, device = *cond.shape[:2], cond.device
        if not exists(lens):
            lens = torch.full((batch, ),
                              cond_seq_len,
                              device=device,
                              dtype=torch.long)

        # text

        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch

        # lens: length of audio prompt
        if exists(text):
            text_lens = (text != -1).sum(dim=-1)
            lens = torch.maximum(
                text_lens, lens
            )  # make sure lengths are at least those of the text characters

        # duration

        cond_mask = lens_to_mask(lens)
        if edit_mask is not None:
            cond_mask = cond_mask & edit_mask

        if isinstance(duration, int):
            duration = torch.full((batch, ),
                                  duration,
                                  device=device,
                                  dtype=torch.long)

        duration = torch.maximum(
            lens + 1, duration)  # just add one token so something is generated
        duration = duration.clamp(max=max_duration)
        max_duration = duration.amax()

        # duplicate test corner for inner time step oberservation
        if duplicate_test:
            test_cond = F.pad(
                cond, (0, 0, cond_seq_len, max_duration - 2 * cond_seq_len),
                value=0.0)

        cond = F.pad(cond, (0, 0, 0, max_duration - cond_seq_len), value=0.0)
        cond_mask = F.pad(cond_mask, (0, max_duration - cond_mask.shape[-1]),
                          value=False)
        cond_mask = cond_mask.unsqueeze(-1)
        step_cond = torch.where(cond_mask, cond, torch.zeros_like(
            cond))  # allow direct control (cut cond audio) with lens passed in

        if batch > 1:
            mask = lens_to_mask(duration)
        else:  # save memory and speed up, as single inference need no mask currently
            mask = None

        # test for no ref audio
        if no_ref_audio:
            cond = torch.zeros_like(cond)

        # neural ode

        def fn(t, x):
            # at each step, conditioning is fixed
            # step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))
            # predict flow
            pred = self.transformer(x=x,
                                    cond=step_cond,
                                    text=text,
                                    time=t,
                                    mask=mask,
                                    drop_audio_cond=False,
                                    drop_text=False)
            if cfg_strength < 1e-5:
                return pred

            null_pred = self.transformer(x=x,
                                         cond=step_cond,
                                         text=text,
                                         time=t,
                                         mask=mask,
                                         drop_audio_cond=True,
                                         drop_text=True)
            return pred + (pred - null_pred) * cfg_strength

        # noise input
        # to make sure batch inference result is same with different batch size, and for sure single inference
        # still some difference maybe due to convolutional layers
        y0 = []
        for dur in duration:
            if exists(seed):
                torch.manual_seed(seed)
            y0.append(
                torch.randn(dur,
                            self.num_channels,
                            device=self.device,
                            dtype=step_cond.dtype))
        y0 = pad_sequence(y0, padding_value=0, batch_first=True)

        if next(self.parameters()).dtype == torch.float16:
            y0 = y0.half()

        t_start = 0

        # duplicate test corner for inner time step oberservation
        if duplicate_test:
            t_start = t_inter
            y0 = (1 - t_start) * y0 + t_start * test_cond
            steps = int(steps * (1 - t_start))

        t = torch.linspace(t_start,
                           1,
                           steps,
                           device=self.device,
                           dtype=step_cond.dtype)
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

        # if next(self.parameters()).dtype == torch.float16:
        #     t = t.half()

        trajectory = odeint(fn, y0, t, **self.odeint_kwargs)

        sampled = trajectory[-1]
        out = sampled
        out = torch.where(cond_mask, cond, out)

        if exists(vocoder):
            out = out.permute(0, 2, 1)
            out = vocoder(out)

        return out, trajectory

    @torch.no_grad()
    def prepare_for_onnx(
            self,
            cond: float["b n d"] | float["b nw"],  # noqa: F722
            text: int["b nt"] | list[str],  # noqa: F722
            duration: int | int["b"],  # noqa: F821
    ):
        batch, device = cond.shape[0], cond.device
        # text
        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch

        if isinstance(duration, int):
            duration = torch.full((batch, ),
                                  duration,
                                  device=device,
                                  dtype=torch.long)

        # raw wave
        if next(self.parameters()).dtype == torch.float16:
            cond = cond.half()
        if cond.ndim == 2:
            cond = self.mel_spec(cond)

            cond = cond.permute(0, 2, 1)
            assert cond.shape[-1] == self.num_channels

        return text, duration, cond

    @torch.no_grad()
    def inference_for_onnx(
        self,
        cond: float["b n d"] | float["b nw"],  # noqa: F722
        text: int["b nt"] | list[str],  # noqa: F722
        duration: int | int["b"],  # noqa: F821
        *,
        lens: int["b"] | None = None,  # noqa: F821
        # ref_audio_len: int["b"] | None = None,  # noqa: F821
        steps=32,
        cfg_strength=1.0,
        sway_sampling_coef=None,
        seed: int | None = None,
        max_duration=4096,
        vocoder: Callable[[float["b d n"]], float["b nw"]]
        | None = None,  # noqa: F722
        no_ref_audio=False,
        duplicate_test=False,
        t_inter=0.1,
        edit_mask=None,
    ):
        self.eval()

        # if next(self.parameters()).dtype == torch.float16:
        #     cond = cond.half()

        # # raw wave

        # if cond.ndim == 2:
        #     cond = self.mel_spec(cond)

        #     cond = cond.permute(0, 2, 1)
        #     assert cond.shape[-1] == self.num_channels

        batch, cond_seq_len, device = *cond.shape[:2], cond.device
        if not exists(lens):
            lens = torch.full((batch, ),
                              cond_seq_len,
                              device=device,
                              dtype=torch.long)

        # lens: length of audio prompt
        # here assume that text is of tensor-input
        if exists(text):
            text_lens = (text != -1).sum(dim=-1)
            lens = torch.maximum(
                text_lens, lens
            )  # make sure lengths are at least those of the text characters

        # duration

        cond_mask = lens_to_mask(lens)
        if edit_mask is not None:
            cond_mask = cond_mask & edit_mask

        # here assume that duraion of tensor-input
        duration = torch.maximum(
            lens + 1, duration)  # just add one token so something is generated
        duration = duration.clamp(max=max_duration)
        max_duration = duration.amax()

        # duplicate test corner for inner time step oberservation
        if duplicate_test:
            test_cond = F.pad(
                cond, (0, 0, cond_seq_len, max_duration - 2 * cond_seq_len),
                value=0.0)

        cond = F.pad(cond, (0, 0, 0, max_duration - cond_seq_len), value=0.0)
        cond_mask = F.pad(cond_mask, (0, max_duration - cond_mask.shape[-1]),
                          value=False)
        cond_mask = cond_mask.unsqueeze(-1)
        step_cond = torch.where(cond_mask, cond, torch.zeros_like(
            cond))  # allow direct control (cut cond audio) with lens passed in

        if batch > 1:
            mask = lens_to_mask(duration)
        else:  # save memory and speed up, as single inference need no mask currently
            mask = None

        # test for no ref audio
        if no_ref_audio:
            cond = torch.zeros_like(cond)

        # neural ode

        def fn(t, x):
            # at each step, conditioning is fixed
            # step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))

            # predict flow
            pred = self.transformer(x=x,
                                    cond=step_cond,
                                    text=text,
                                    time=t,
                                    mask=mask,
                                    drop_audio_cond=False,
                                    drop_text=False)
            if cfg_strength < 1e-5:
                return pred

            null_pred = self.transformer(x=x,
                                         cond=step_cond,
                                         text=text,
                                         time=t,
                                         mask=mask,
                                         drop_audio_cond=True,
                                         drop_text=True)
            return pred + (pred - null_pred) * cfg_strength

        # noise input
        # to make sure batch inference result is same with different batch size, and for sure single inference
        # still some difference maybe due to convolutional layers
        y0 = []
        for dur in duration:
            if exists(seed):
                torch.manual_seed(seed)
            y0.append(
                torch.randn(dur,
                            self.num_channels,
                            device=self.device,
                            dtype=step_cond.dtype))
        y0 = pad_sequence(y0, padding_value=0, batch_first=True)

        t_start = 0

        # duplicate test corner for inner time step oberservation
        if duplicate_test:
            t_start = t_inter
            y0 = (1 - t_start) * y0 + t_start * test_cond
            steps = int(steps * (1 - t_start))

        t = torch.linspace(t_start,
                           1,
                           steps,
                           device=self.device,
                           dtype=step_cond.dtype)
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

        trajectory = odeint(fn, y0, t, **self.odeint_kwargs)

        sampled = trajectory[-1]
        out = sampled
        out = torch.where(cond_mask, cond, out)

        # assume that batch=1 for inference
        # print('ref_audio_len: ', ref_audio_len)
        # ref_audio_len = ref_audio_len[0]
        # if exists(vocoder):
        #     out = out.permute(0, 2, 1)
        #     out = vocoder(out)
        # generated = out.to(torch.float32)
        # generated = generated[:, ref_audio_len:, :]
        # generated_mel_spec = generated.permute(0, 2, 1)
        # # generated_wave = vocoder.decode(generated_mel_spec.cpu())
        # generated_wave = vocoder.decode(generated_mel_spec)
        # if rms < target_rms:
        #     generated_wave = generated_wave * rms / target_rms
        # print('generated_wave size: ', generated_wave.size())
        # wav -> numpy
        # generated_wave = generated_wave.squeeze().cpu().numpy()
        # return generated_wave, generated_mel_spec
        return out, trajectory

    def vocoding_for_onnx(self, out, vocoder, ref_audio_len):
        # assume that batch=1 for inference
        print('ref_audio_len: ', ref_audio_len)
        ref_audio_len = ref_audio_len[0]
        # if exists(vocoder):
        #     out = out.permute(0, 2, 1)
        #     out = vocoder(out)
        generated = out.to(torch.float32)
        generated = generated[:, ref_audio_len:, :]
        generated_mel_spec = generated.permute(0, 2, 1)
        # generated_wave = vocoder.decode(generated_mel_spec.cpu())
        generated_wave = vocoder.decode(generated_mel_spec.cpu())
        # if rms < target_rms:
        #     generated_wave = generated_wave * rms / target_rms
        print('generated_wave size: ', generated_wave.size())
        # wav -> numpy
        # generated_wave = generated_wave.squeeze().cpu().numpy()
        return generated_wave, generated_mel_spec

    def forward(
        self,
        inp: float["b n d"] | float["b nw"],  # mel or raw wave  # noqa: F722
        text: int["b nt"] | list[str],  # noqa: F722
        *,
        lens: int["b"] | None = None,  # noqa: F821
        noise_scheduler: str | None = None,
    ):
        # handle raw wave
        if inp.ndim == 2:
            # now better avoid directly loading raw wave
            assert 0==1
            inp = self.mel_spec(inp)
            inp = inp.permute(0, 2, 1)
            assert inp.shape[-1] == self.num_channels

        batch, seq_len, dtype, device, _σ1 = *inp.shape[:
                                                        2], inp.dtype, self.device, self.sigma

        # handle text as string
        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch

        # lens and mask
        if not exists(lens):
            lens = torch.full((batch, ), seq_len, device=device)

        mask = lens_to_mask(
            lens, length=seq_len
        )  # useless here, as collate_fn will pad to max length in batch

        # get a random span to mask out for training conditionally
        frac_lengths = torch.zeros(
            (batch, ),
            device=self.device).float().uniform_(*self.frac_lengths_mask)
        rand_span_mask = mask_from_frac_lengths(lens, frac_lengths)

        if exists(mask):
            rand_span_mask &= mask

        # mel is x1
        x1 = inp

        # x0 is gaussian noise
        x0 = torch.randn_like(x1)

        # time step
        time = torch.rand((batch, ), dtype=dtype, device=self.device)
        # TODO. noise_scheduler

        # sample xt (φ_t(x) in the paper)
        t = time.unsqueeze(-1).unsqueeze(-1)
        φ = (1 - t) * x0 + t * x1
        flow = x1 - x0

        # only predict what is within the random mask span for infilling
        cond = torch.where(rand_span_mask[..., None], torch.zeros_like(x1), x1)

        # transformer and cfg training with a drop rate
        drop_audio_cond = random() < self.audio_drop_prob  # p_drop in voicebox paper
        drop_audio_text_cond = random() < self.cond_drop_prob # p_uncond in voicebox paper

        if drop_audio_text_cond: 
            drop_audio_cond = True
            drop_text = True
        else:
            drop_text = False

        # Three mode:
        # 1. p_uncond, audio and text off
        # 2. p_cond, audio p_drop off
        # 3. 0.7~1.0, audio continious span off

        # if want rigourously mask out padding, record in collate_fn in dataset.py, and pass in here
        # adding mask will use more memory, thus also need to adjust batchsampler with scaled down threshold for long sequences
        pred = self.transformer(x=φ,
                                cond=cond,
                                text=text,
                                time=time,
                                drop_audio_cond=drop_audio_cond,
                                drop_text=drop_text)

        # flow matching loss
        loss = F.mse_loss(pred, flow, reduction="none")
        if drop_audio_cond or drop_audio_text_cond:
            loss = loss[mask]
        else:
            loss = loss[rand_span_mask]

        return loss.mean(), cond, pred
