import os
from enum import Enum
from typing import Dict

import requests  # type: ignore[import]
from requests import HTTPError

from .base import Backend


class Endpoints(str, Enum):
    COMPLETIONS = "https://api.anthropic.com/v1/complete"


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
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "x-api-key": api_key,
        }

        assert api_key is not None
        return headers
