import os
import warnings
from enum import Enum
from typing import Any, Dict, Iterable, List, Sized, Tuple

import requests  # type: ignore[import]
import srsly  # type: ignore[import]
from requests import HTTPError

from ....compat import Literal
from ..base import REST


class Endpoints(str, Enum):
    COMPLETIONS = "https://api.anthropic.com/v1/complete"


class SystemPrompt(str, Enum):
    """Specifies the system prompt for Claude
    c.f. https://console.anthropic.com/docs/prompt-design#what-is-a-prompt
    """

    HUMAN = "\n\nHuman:"
    ASST = "\n\nAssistant:"


class Anthropic(REST):
    MODEL_NAMES = {
        "claude-1": Literal["claude-1", "claude-1-100k"],
    }

    @property
    def credentials(self) -> Dict[str, str]:
        # Fetch and check the key, set up headers
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key is None:
            warnings.warn(
                "Could not find the API key to access the Anthropic Claude API. Ensure you have an API key "
                "set up via the Anthropic console (https://console.anthropic.com/), then make it available as "
                "an environment variable 'ANTHROPIC_API_KEY."
            )

        return {"X-API-Key": api_key if api_key else ""}

    def _verify_auth(self) -> None:
        # Execute a dummy prompt. If the API setup is incorrect, we should fail at initialization time.
        try:
            self(["test"])
        except ValueError as err:
            if "authentication_error" in str(err):
                warnings.warn(
                    "Authentication with provided API key failed. Please double-check you provided the correct "
                    "credentials."
                )
            else:
                raise err

    def __call__(self, prompts: Iterable[str]) -> Iterable[str]:
        headers = {
            **self._credentials,
            "model": self._name,
            "Content-Type": "application/json",
        }

        api_responses: List[str] = []
        prompts = list(prompts)

        def _request(json_data: Dict[str, Any]) -> Dict[str, Any]:
            r = self.retry(
                call_method=requests.post,
                url=self._endpoint,
                headers=headers,
                json={**json_data, **self._config, "model": self._name},
                timeout=self._max_request_time,
            )
            try:
                r.raise_for_status()
            except HTTPError as ex:
                res_content = srsly.json_loads(r.content.decode("utf-8"))
                # Include specific error message in exception.
                error = res_content.get("error", {})
                error_msg = f"Request to Anthropic API failed: {error}"
                if error["type"] == "not_found_error":
                    error_msg += f". Ensure that the selected model ({self._name}) is supported by the API."
                raise ValueError(error_msg) from ex
            response = r.json()

            # c.f. https://console.anthropic.com/docs/api/errors
            if "error" in response:
                if self._strict:
                    raise ValueError(f"API call failed: {response}.")
                else:
                    assert isinstance(prompts, Sized)
                    return {"error": [srsly.json_dumps(response)] * len(prompts)}
            return response

        # Anthropic API currently doesn't accept batch prompts, so we're making
        # a request for each iteration. This approach can be prone to rate limit
        # errors. In practice, you can adjust _max_request_time so that the
        # timeout is larger.
        responses = [
            _request({"prompt": f"{SystemPrompt.HUMAN} {prompt}{SystemPrompt.ASST}"})
            for prompt in prompts
        ]

        for response in responses:
            if "completion" in response:
                api_responses.append(response["completion"])
            else:
                api_responses.append(srsly.json_dumps(response))

        assert len(api_responses) == len(prompts)
        return api_responses

    @classmethod
    def get_model_names(cls) -> Tuple[str, ...]:
        return (
            # claude-1
            "claude-1",
            "claude-1-100k",
            # claude-instant-1
            "claude-instant-1",
            "claude-instant-1-100k",
            # claude-instant-1.1
            "claude-instant-1.1",
            "claude-instant-1.1-100k",
            # claude-1.3
            "claude-1.3",
            "claude-1.3-100k",
            # others
            "claude-1.0",
            "claude-1.2",
        )
