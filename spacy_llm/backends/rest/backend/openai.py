import os
from typing import Set, Dict, Iterable, List

import requests  # type: ignore[import]
import srsly  # type: ignore[import]

from .base import Backend


class OpenAIBackend(Backend):
    @property
    def _default_endpoint(self) -> str:
        return "https://api.openai.com/v1/completions"

    @property
    def supported_models(self) -> Set[str]:
        return {
            "text-davinci-003",
            "text-davinci-002",
            "text-curie-001",
            "text-babbage-001",
            "text-ada-001",
            "davinci",
            "curie",
            "babbage",
            "ada",
        }

    @property
    def credentials(self) -> Dict[str, str]:
        model = self._config.get("model", "text-davinci-003")
        # Fetch and check the key
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError(
                "Could not find the API key to access the openai API. Ensure you have an API key "
                "set up via https://beta.openai.com/account/api-keys, then make it available as "
                "an environment variable 'PRODIGY_OPENAI_KEY', for instance in a .env file."
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
                "Visit https://beta.openai.com/account/api-keys to check your API keys."
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

        r = self.retry(
            lambda: requests.post(
                self._url,
                headers=headers,
                json={"prompt": prompts, **self._config},
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
            if "text" in response:
                api_responses.append(response["text"])
            else:
                api_responses.append(srsly.json_dumps(response))

        return api_responses
