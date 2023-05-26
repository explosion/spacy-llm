import time
from typing import Dict, Iterable

from .base import Backend

_NOOP_RESPONSE = ""


class NoOpBackend(Backend):
    """NoOp backend. Used for tests."""

    _CALL_TIMEOUT = 0.01

    @property
    def supported_models(self) -> Dict[str, str]:
        return {"NoOp": ""}

    @property
    def credentials(self) -> Dict[str, str]:
        return {}

    def __call__(self, prompts: Iterable[str]) -> Iterable[str]:
        # Assume time penalty for API calls.
        time.sleep(NoOpBackend._CALL_TIMEOUT)
        return [_NOOP_RESPONSE] * len(list(prompts))
