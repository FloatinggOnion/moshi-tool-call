from __future__ import annotations

import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import sentencepiece
import sphn
from huggingface_hub import hf_hub_download

from moshi_mlx import models
from moshi_mlx.models.tts import (
    DEFAULT_DSM_TTS_REPO,
    DEFAULT_DSM_TTS_VOICE_REPO,
    TTSModel,
)


os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")


@dataclass(slots=True)
class SpeechRenderResult:
    text: str
    wav_path: Path
    duration_seconds: float


class MoshiSpeechInjector:
    def __init__(
        self,
        hf_repo: str = DEFAULT_DSM_TTS_REPO,
        voice_repo: str = DEFAULT_DSM_TTS_VOICE_REPO,
        voice_name: str = "alba-mackenna/announcer.wav",
        cfg_coef: float = 2.0,
        temp: float = 0.6,
        nq: int = 32,
        max_padding: int = 8,
        initial_padding: int = 2,
        final_padding: int = 4,
        padding_between: int = 1,
    ) -> None:
        self.hf_repo = hf_repo
        self.voice_repo = voice_repo
        self.voice_name = voice_name
        self.cfg_coef = cfg_coef
        self.temp = temp
        self.nq = nq
        self.max_padding = max_padding
        self.initial_padding = initial_padding
        self.final_padding = final_padding
        self.padding_between = padding_between

        self._config_path = hf_hub_download(hf_repo, "config.json")
        self._config = json.loads(Path(self._config_path).read_text(encoding="utf-8"))
        self._mimi_path = hf_hub_download(hf_repo, self._config["mimi_name"])
        self._moshi_path = hf_hub_download(hf_repo, self._config.get("moshi_name", "model.safetensors"))
        self._tokenizer_path = hf_hub_download(hf_repo, self._config["tokenizer_name"])

        self._lm_config = models.LmConfig.from_config_dict(self._config)
        self._model = models.Lm(self._lm_config)
        self._model.set_dtype(mx.bfloat16)
        self._model.load_pytorch_weights(self._moshi_path, self._lm_config, strict=True)

        self._tokenizer = sentencepiece.SentencePieceProcessor(self._tokenizer_path)  # type: ignore[arg-type]
        self._audio_tokenizer = models.mimi.Mimi(models.mimi_202407(self._lm_config.generated_codebooks))
        self._audio_tokenizer.load_pytorch_weights(self._mimi_path, strict=True)

        self._tts = TTSModel(
            self._model,
            self._audio_tokenizer,
            self._tokenizer,
            voice_repo=self.voice_repo,
            n_q=self.nq,
            temp=self.temp,
            cfg_coef=self.cfg_coef,
            max_padding=self.max_padding,
            initial_padding=self.initial_padding,
            final_padding=self.final_padding,
            padding_bonus=0.0,
            raw_config=self._config,
        )

        if self._tts.valid_cfg_conditionings:
            self._cfg_coef_conditioning: float | None = self._tts.cfg_coef
            self._tts.cfg_coef = 1.0
            self._cfg_is_no_text = False
            self._cfg_is_no_prefix = False
        else:
            self._cfg_coef_conditioning = None
            self._cfg_is_no_text = True
            self._cfg_is_no_prefix = True

    @property
    def sample_rate(self) -> int:
        return self._tts.mimi.sample_rate

    def render_pcm(
        self,
        text: str,
        voice_name: str | None = None,
    ) -> np.ndarray:
        voice_name = voice_name or self.voice_name
        entries = self._tts.prepare_script([text], padding_between=self.padding_between)
        voice_path = self._tts.get_voice_path(voice_name)
        attributes = [
            self._tts.make_condition_attributes([voice_path], cfg_coef=self._cfg_coef_conditioning)
        ]
        prefixes = None
        if not self._tts.multi_speaker:
            prefixes = [self._tts.get_prefix(voice_path)]

        result = self._tts.generate(
            [entries],
            attributes,
            prefixes=prefixes,
            cfg_is_no_prefix=self._cfg_is_no_prefix,
            cfg_is_no_text=self._cfg_is_no_text,
        )

        wav_frames = []
        for frame in result.frames:
            wav_frames.append(self._tts.mimi.decode_step(frame))

        wav = mx.concat(wav_frames, axis=-1)
        if result.end_steps[0] is None:
            wav_length = wav.shape[-1]
        else:
            wav_length = int((self._tts.mimi.sample_rate * (result.end_steps[0] + self._tts.final_padding) / self._tts.mimi.frame_rate))

        clipped = mx.clip(wav[0, :, :wav_length], -1, 1)
        return np.array(clipped)[0]

    def render(
        self,
        text: str,
        output_path: Path,
        voice_name: str | None = None,
    ) -> SpeechRenderResult:
        pcm = self.render_pcm(text, voice_name)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sphn.write_wav(output_path, pcm, self.sample_rate)

        return SpeechRenderResult(
            text=text,
            wav_path=output_path,
            duration_seconds=pcm.shape[-1] / self.sample_rate,
        )
