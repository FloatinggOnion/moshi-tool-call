from __future__ import annotations

from dataclasses import dataclass
import re
import time
from typing import Any


_BOUNDARY_RE = re.compile(r"[.!?](?:['\")\]]+)?(?:\s|$)")


@dataclass(slots=True)
class TranscriptChunk:
    text: str
    reason: str
    emitted_at: float
    final: bool = False


class RollingTranscriptBuffer:
    def __init__(self, stability_window_seconds: float = 0.7, min_chunk_chars: int = 12) -> None:
        self.stability_window_seconds = stability_window_seconds
        self.min_chunk_chars = min_chunk_chars
        self._snapshot = ""
        self._committed = ""
        self._last_update = 0.0

    @staticmethod
    def _longest_common_prefix(left: str, right: str) -> str:
        limit = min(len(left), len(right))
        index = 0
        while index < limit and left[index] == right[index]:
            index += 1
        return left[:index]

    def observe(self, snapshot: str, now: float | None = None) -> list[TranscriptChunk]:
        current_time = now if now is not None else time.monotonic()
        snapshot = snapshot.rstrip()

        if snapshot != self._snapshot:
            self._snapshot = snapshot
            self._last_update = current_time

        return self.flush(now=current_time)

    def flush(self, now: float | None = None, final: bool = False) -> list[TranscriptChunk]:
        current_time = now if now is not None else time.monotonic()
        if not self._snapshot:
            return []

        if not final and (current_time - self._last_update) < self.stability_window_seconds:
            return []

        snapshot = self._snapshot
        committed = self._committed

        if snapshot.startswith(committed):
            tail = snapshot[len(committed):]
        else:
            committed = self._longest_common_prefix(committed, snapshot)
            self._committed = committed
            tail = snapshot[len(committed):]

        if not tail:
            return []

        emit_text = ""
        reason = "stable_boundary"

        if final:
            emit_text = tail
            reason = "final_flush"
        else:
            matches = list(_BOUNDARY_RE.finditer(tail))
            if matches:
                emit_end = matches[-1].end()
                emit_text = tail[:emit_end].rstrip()
            else:
                return []

        if len(emit_text.strip()) < self.min_chunk_chars and not final:
            return []

        emit_text = emit_text.strip()
        if not emit_text:
            return []

        self._committed = committed + emit_text
        return [
            TranscriptChunk(
                text=emit_text,
                reason=reason,
                emitted_at=current_time,
                final=final,
            )
        ]

    def reset(self) -> None:
        self._snapshot = ""
        self._committed = ""
        self._last_update = 0.0

    @property
    def snapshot(self) -> str:
        return self._snapshot

    @property
    def committed(self) -> str:
        return self._committed

    @property
    def pending(self) -> str:
        if self._snapshot.startswith(self._committed):
            return self._snapshot[len(self._committed):]
        return self._snapshot
