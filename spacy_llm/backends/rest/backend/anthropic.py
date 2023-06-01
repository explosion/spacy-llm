import os
from enum import Enum
from typing import Any, Dict, Iterable, List, Sized

import requests  # type: ignore[import]
import srsly  # type: ignore[import]
from requests import HTTPError

from .base import Backend


class Endpoints(str, Enum):
    COMPLETIONS = "https://api.anthropic.com/v1/complete"


class Speaker(str, Enum):
    """Specifies the prompt prefix for engineering Claude-style prompts
    c.f. https://console.anthropic.com/docs/prompt-design#what-is-a-prompt
    """

    HUMAN = "\n\nHuman:"
    ASST = "\n\nAssistant:"


class AnthropicBackend(Backend):
    @property
    def supported_models(self) -> Dict[str, str]:
        """Returns supported models with their endpoints.
        Based on https://console.anthropic.com/docs/api/reference
        RETURNS (Dict[str, str]): Supported models with their endpoints.
        """
        return {
            "claude-v1": Endpoints.COMPLETIONS.value,
            "claude-v1-100k": Endpoints.COMPLETIONS.value,
            "claude-instant-v1": Endpoints.COMPLETIONS.value,
            "claude-instant-v1-100k": Endpoints.COMPLETIONS.value,
            # sub-versions of the models
            "claude-v1.3": Endpoints.COMPLETIONS.value,
            "claude-v1.3-100k": Endpoints.COMPLETIONS.value,
            "claude-v1.2": Endpoints.COMPLETIONS.value,
            "claude-v1.0": Endpoints.COMPLETIONS.value,
            "claude-instant-v1.1": Endpoints.COMPLETIONS.value,
            "claude-instant-v1.1-100k": Endpoints.COMPLETIONS.value,
            "claude-instant-v1.0": Endpoints.COMPLETIONS.value,
        }

    @property
    def credentials(self) -> Dict[str, str]:
        # Fetch and check the key
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key is None:
            raise ValueError(
                "Could not find the API key to access the Anthropic Claude API. Ensure you have an API key "
                "set up via the Anthropic console (https://console.anthropic.com/), then make it available as "
                "an environment variable 'ANTHROPIC_API_KEY."
            )

        # Ensure model is supported
        model = self._config["model"]
        if model not in self.supported_models:
            raise ValueError(
                f"The specified model '{model}' is not supported by the /v1/completions endpoint. "
                f"Choices are: {sorted(list(self.supported_models))} ."
                "(See the Anthropic API documentation: https://console.anthropic.com/docs/api/reference)"
            )

        # Set-up headers
        headers = {"X-API-Key": api_key}
        assert api_key is not None
        return headers

    def __call__(self, prompts: Iterable[str]) -> Iterable[str]:
        headers = {
            **self._credentials,
            "Content-Type": "application/json",
        }

        api_responses: List[str] = []
        prompts = list(prompts)
        url = self._url if self._url else self.supported_models[self._config["model"]]

        def _request(json_data: Dict[str, Any]) -> Dict[str, Any]:
            r = self.retry(
                call_method=requests.post,
                url=url,
                headers=headers,
                json={**json_data, **self._config},
                timeout=self._max_request_time,
            )

            try:
                r.raise_for_status()
            except HTTPError as ex:
                res_content = srsly.json_loads(r.content.decode("utf-8"))
                # Include specific error message in exception.
                raise ValueError(
                    f"Request to Anthropic API failed: {res_content.get('error', {})}"
                ) from ex
            responses = r.json()

            # c.f. https://console.anthropic.com/docs/api/errors
            if "error" in responses:
                if self._strict:
                    raise ValueError(f"API call failed: {responses}.")
                else:
                    assert isinstance(prompts, Sized)
                    return {"error": [srsly.json_dumps(responses)] * len(prompts)}
            return responses

        responses = [
            _request({"prompt": f"{Speaker.HUMAN}{prompt}{Speaker.ASST}"})
            for prompt in prompts
        ]

        if "error" in responses:
            return responses["error"]
        assert len(responses["choices"]) == len(prompts)

        for response in responses:
            if "completion" in response:
                api_responses.append(response["completion"])
            else:
                api_responses.append(srsly.json_dumps(response))
        return api_responses
