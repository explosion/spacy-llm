import os
from enum import Enum
from typing import Set, Dict, Iterable, List, Tuple, Any, Union

import requests  # type: ignore[import]
import srsly  # type: ignore[import]

from .base import Backend


class Endpoints(str, Enum):
    chat = "https://api.openai.com/v1/chat/completions"
    completions = "https://api.openai.com/v1/completions"


class OpenAIBackend(Backend):
    def __init__(
        self,
        config: Dict[Any, Any],
        strict: bool,
        max_tries: int,
        timeout: int,
    ):
        super().__init__(
            config=config, strict=strict, max_tries=max_tries, timeout=timeout
        )
        self._check_api_endpoint_compatibility()

    @property
    def _default_endpoint(self) -> str:
        return Endpoints.completions.value

    @property
    def supported_models(self) -> Set[str]:
        return set(OpenAIBackend._supported_models_with_endpoints().keys())

    @staticmethod
    def _supported_models_with_endpoints() -> Dict[str, Tuple[str, ...]]:
        """Returns supported models with their endpoints.
        RETURNS (Dict[str, Tuple[str, ...]]): Supported models with their endpoints.
        """
        return {
            "gpt-4": (Endpoints.chat,),
            "gpt-4-0314": (Endpoints.chat,),
            "gpt-4-32k": (Endpoints.chat,),
            "gpt-4-32k-0314": (Endpoints.chat,),
            "gpt-3.5-turbo": (Endpoints.chat,),
            "gpt-3.5-turbo-0301": (Endpoints.chat,),
            "text-davinci-003": (Endpoints.completions,),
            "text-davinci-002": (Endpoints.completions,),
            "text-curie-001": (Endpoints.completions,),
            "text-babbage-001": (Endpoints.completions,),
            "text-ada-001": (Endpoints.completions,),
            "davinci": (Endpoints.completions,),
            "curie": (Endpoints.completions,),
            "babbage": (Endpoints.completions,),
            "ada": (Endpoints.completions,),
        }

    def _check_api_endpoint_compatibility(self):
        """Checks whether specified model supports the supported API endpoint."""
        model_endpoints = OpenAIBackend._supported_models_with_endpoints()[
            self._config["model"]
        ]
        if self._url not in model_endpoints:
            raise ValueError(
                f"Specified model {self._config['model']} supports of the following endpoints: "
                f"{', '.join(model_endpoints)}. However, endpoint {self._url} has been configured. Please ensure that "
                f"model and endpoint match."
            )

    @property
    def credentials(self) -> Dict[str, str]:
        model = self._config.get("model", "text-davinci-003")
        # Fetch and check the key
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError(
                "Could not find the API key to access the OpenAI API. Ensure you have an API key "
                "set up via https://platform.openai.com/account/api-keys, then make it available as "
                "an environment variable 'OPENAI_API_KEY', for instance in a .env file."
            )

        # Check the access and get a list of available models to verify the model argument (if not None)
        # Even if the model is None, this call is used as a healthcheck to verify access.
        headers = {"Authorization": f"Bearer {api_key}"}
        r = self.retry(
            lambda: requests.get(
                "https://api.openai.com/v1/models",
                headers=headers,
            ),
        )
        if r.status_code == 422:
            raise ValueError(
                "Could not access api.openai.com -- 422 permission denied."
                "Visit https://platform.openai.com/account/api-keys to check your API keys."
            )
        elif r.status_code != 200:
            raise ValueError(
                "Error accessing api.openai.com" f"{r.status_code}: {r.text}"
            )

        response = r.json()["data"]
        models = [response[i]["id"] for i in range(len(response))]
        if model not in models:
            raise ValueError(
                f"The specified model '{model}' is not available. Choices are: {sorted(set(models))}"
            )

        if model not in self.supported_models:
            raise ValueError(
                f"The specified model '{model}' is not supported by the /v1/completions endpoint. "
                f"Choices are: {sorted(list(self.supported_models))} ."
                "(See OpenAI API documentation: https://platform.openai.com/docs/models/gpt-3)"
            )

        assert api_key is not None
        return headers

    def __call__(self, prompts: Iterable[str]) -> Iterable[str]:
        headers = {
            **self._credentials,
            "Content-Type": "application/json",
        }
        api_responses: List[str] = []
        prompts = list(prompts)

        data: Dict[str, Union[List[str], List[Dict[str, str]]]] = {}
        if self._url == Endpoints.chat:
            data = {
                "messages": [{"role": "user", "content": prompt} for prompt in prompts]
            }
        elif self._url == Endpoints.completions:
            data = {"prompt": prompts}

        r = self.retry(
            lambda: requests.post(
                self._url,
                headers=headers,
                json={**data, **self._config},
                timeout=self._timeout,
            ),
        )
        r.raise_for_status()
        responses = r.json()

        # Process responses.
        if "error" in responses:
            if self._strict:
                raise ValueError(f"API call failed: {responses}.")
            else:
                return [srsly.json_dumps(responses)] * len(prompts)
        assert len(responses["choices"]) == len(prompts)

        for prompt, response in zip(prompts, responses["choices"]):
            if self._url.endswith("chat"):
                if "message" in response:
                    api_responses.append(response["message"]["content"])
                else:
                    api_responses.append(srsly.json_dumps(response))
            elif self._url.endswith("completions"):
                if "text" in response:
                    api_responses.append(response["text"])
                else:
                    api_responses.append(srsly.json_dumps(response))

        return api_responses
