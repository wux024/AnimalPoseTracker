"""Structured progress events shared by the trainer, CLI and GUI."""

import json
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Optional, TextIO


@dataclass
class TrainingEvent:
    """One machine-readable training status update."""

    event: str
    epoch: Optional[int] = None
    epochs: Optional[int] = None
    step: Optional[int] = None
    steps: Optional[int] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    message: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, allow_nan=False)


class EventEmitter:
    """Send progress to an optional callback and/or a JSON-lines stream."""

    def __init__(
        self,
        callback: Optional[Callable[[TrainingEvent], None]] = None,
        stream: Optional[TextIO] = None,
    ) -> None:
        self.callback = callback
        self.streams = [stream] if stream is not None else []

    def add_stream(self, stream: TextIO) -> None:
        if stream is not None:
            self.streams.append(stream)

    def emit(self, event: TrainingEvent) -> None:
        if self.callback is not None:
            self.callback(event)
        line = event.to_json() + "\n"
        for stream in self.streams:
            stream.write(line)
            stream.flush()

    @classmethod
    def json_stdout(
        cls,
        callback: Optional[Callable[[TrainingEvent], None]] = None,
    ) -> "EventEmitter":
        return cls(callback=callback, stream=sys.stdout if sys.stdout is not None else None)
