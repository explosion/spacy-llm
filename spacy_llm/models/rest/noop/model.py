import time
from typing import Dict, Iterable, Tuple

from ..base import REST

_NOOP_RESPONSE = ""


class NoOpModel(REST):
    """NoOp model. Used for tests only."""

    _CALL_TIMEOUT = 0.01

    def __init__(self):
        super().__init__(
            name="NoOp",
            endpoint="NoOp",
            config={},
            strict=True,
            max_tries=1,
            interval=1,
            max_request_time=1,
        )

    @property
    def credentials(self) -> Dict[str, str]:
        return {}

    def _verify_auth(self) -> None:
        pass

    def __call__(self, prompts: Iterable[str]) -> Iterable[str]:
        # Assume time penalty for API calls.
        time.sleep(NoOpModel._CALL_TIMEOUT)
        return [_NOOP_RESPONSE] * len(list(prompts))

    @classmethod
    def get_model_names(cls) -> Tuple[str, ...]:
        return ("NoOp",)
