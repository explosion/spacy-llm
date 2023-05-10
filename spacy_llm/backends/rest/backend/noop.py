import time
from typing import Set, Dict, Iterable

from .base import Backend


class NoOpBackend(Backend):
    """NoOp backend. Used for tests."""

    _CALL_TIMEOUT = 0.01

    @property
    def _default_endpoint(self) -> str:
        return ""

    @property
    def supported_models(self) -> Set[str]:
        return set()

    @property
    def credentials(self) -> Dict[str, str]:
        return {}

    def __call__(self, prompts: Iterable[str]) -> Iterable[str]:
        # Assume time penalty for API calls.
        time.sleep(NoOpBackend._CALL_TIMEOUT)
        return [""] * len(list(prompts))
