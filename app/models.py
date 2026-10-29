"""Shared data types for jobs and transcripts."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

Status = Literal[
    "awaiting_upload", "queued", "processing", "done", "failed", "cancelled"
]
Stage = Literal["convert", "diarize", "transcribe", "merge"]

LANG_SPECIAL = {"auto", "multi"}
MODELS = {"small", "medium", "large-v3-turbo"}
DIARIZATION = {"pyannote", "sherpa", "none"}


@dataclass
class JobParams:
    language: str = "en"
    model: str = "large-v3-turbo"
    diarization: str = "pyannote"
    num_speakers: Optional[int] = None
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None
    prompt: Optional[str] = None
    title: Optional[str] = None

    def validate(self) -> None:
        if self.model not in MODELS:
            raise ValueError(f"model must be one of {sorted(MODELS)}")
        if self.diarization not in DIARIZATION:
            raise ValueError(f"diarization must be one of {sorted(DIARIZATION)}")
        # language: ISO-ish code, or a special mode. Keep it permissive.
        if not self.language or len(self.language) > 16:
            raise ValueError("language is required")
        for n in (self.num_speakers, self.min_speakers, self.max_speakers):
            if n is not None and (n < 1 or n > 50):
                raise ValueError("speaker counts must be in 1..50")

    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "model": self.model,
            "diarization": self.diarization,
            "num_speakers": self.num_speakers,
            "min_speakers": self.min_speakers,
            "max_speakers": self.max_speakers,
            "prompt": self.prompt,
            "title": self.title,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "JobParams":
        return cls(**{k: d.get(k) for k in cls().__dict__ if k in d})


# --- pipeline intermediate types ---

@dataclass
class Word:
    start: float
    end: float
    text: str
    lang: Optional[str] = None


@dataclass
class Turn:
    start: float
    end: float
    speaker: str


@dataclass
class Utterance:
    speaker: str
    start: float
    end: float
    text: str
    lang: Optional[str] = None
