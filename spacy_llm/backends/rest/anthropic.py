import os
from enum import Enum
from typing import Any, Dict, Iterable, List, Sized

import requests  # type: ignore[import]
import srsly  # type: ignore[import]
from requests import HTTPError

from .base import Backend


class Endpoints(str, Enum):
    COMPLETIONS = "https://api.anthropic.com/v1/complete"


class SystemPrompt(str, Enum):
    """Specifies the system prompt for Claude
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
            "claude-1": Endpoints.COMPLETIONS.value,
            "claude-1-100k": Endpoints.COMPLETIONS.value,
            "claude-instant-1": Endpoints.COMPLETIONS.value,
            "claude-instant-1-100k": Endpoints.COMPLETIONS.value,
            # sub-versions of the models
            "claude-1.3": Endpoints.COMPLETIONS.value,
            "claude-1.3-100k": Endpoints.COMPLETIONS.value,
            "claude-1.2": Endpoints.COMPLETIONS.value,
            "claude-1.0": Endpoints.COMPLETIONS.value,
            "claude-instant-1.1": Endpoints.COMPLETIONS.value,
            "claude-instant-1.1-100k": Endpoints.COMPLETIONS.value,
            "claude-instant-1.0": Endpoints.COMPLETIONS.value,
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
                f"The specified model '{model}' is not supported by the /v1/complete endpoint. "
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
