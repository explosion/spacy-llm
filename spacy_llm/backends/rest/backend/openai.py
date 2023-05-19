import os
from enum import Enum
from typing import Dict, Iterable, List, Any, Sized

import requests  # type: ignore[import]
import srsly  # type: ignore[import]

from .base import Backend


class Endpoints(str, Enum):
    chat = "https://api.openai.com/v1/chat/completions"
    non_chat = "https://api.openai.com/v1/completions"


class OpenAIBackend(Backend):
    @property
    def supported_models(self) -> Dict[str, str]:
        """Returns supported models with their endpoints.
        RETURNS (Dict[str, str]): Supported models with their endpoints.
        """
        return {
            "gpt-4": Endpoints.chat.value,
            "gpt-4-0314": Endpoints.chat.value,
            "gpt-4-32k": Endpoints.chat.value,
            "gpt-4-32k-0314": Endpoints.chat.value,
            "gpt-3.5-turbo": Endpoints.chat.value,
            "gpt-3.5-turbo-0301": Endpoints.chat.value,
            "text-davinci-003": Endpoints.non_chat.value,
            "text-davinci-002": Endpoints.non_chat.value,
            "text-curie-001": Endpoints.non_chat.value,
            "text-babbage-001": Endpoints.non_chat.value,
            "text-ada-001": Endpoints.non_chat.value,
            "davinci": Endpoints.non_chat.value,
            "curie": Endpoints.non_chat.value,
            "babbage": Endpoints.non_chat,
            "ada": Endpoints.non_chat.value,
        }

    @property
    def credentials(self) -> Dict[str, str]:
        model = self._config["model"]
        # Fetch and check the key
        api_key = os.getenv("OPENAI_API_KEY")
        api_org = os.getenv("OPENAI_API_ORG")
        if api_key is None:
            raise ValueError(
                "Could not find the API key to access the OpenAI API. Ensure you have an API key "
                "set up via https://platform.openai.com/account/api-keys, then make it available as "
                "an environment variable 'OPENAI_API_KEY."
            )

        # Check the access and get a list of available models to verify the model argument (if not None)
        # Even if the model is None, this call is used as a healthcheck to verify access.
        headers = {
            "Authorization": f"Bearer {api_key}",
        }
        if api_org:
            headers["OpenAI-Organization"] = api_org
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
        url = self._url if self._url else self.supported_models[self._config["model"]]

        def _request(json_data: Dict[str, Any]) -> Dict[str, Any]:
            r = self.retry(
                lambda: requests.post(
                    url,
                    headers=headers,
                    json={**json_data, **self._config},
                    timeout=self._timeout,
                ),
            )
            r.raise_for_status()
            responses = r.json()

            if "error" in responses:
                if self._strict:
                    raise ValueError(f"API call failed: {responses}.")
                else:
                    assert isinstance(prompts, Sized)
                    return {"error": [srsly.json_dumps(responses)] * len(prompts)}

            return responses

        if url == Endpoints.chat:
            # The OpenAI API doesn't support batching for /chat/completions yet, so we have to send individual requests.
            for prompt in prompts:
                responses = _request(
                    {"messages": [{"role": "user", "content": prompt}]}
                )
                if "error" in responses:
                    return responses["error"]

                # Process responses.
                assert len(responses["choices"]) == 1
                response = responses["choices"][0]
                api_responses.append(
                    response.get("message", {}).get(
                        "content", srsly.json_dumps(response)
                    )
                )

        elif url == Endpoints.non_chat:
            responses = _request({"prompt": prompts})
            if "error" in responses:
                return responses["error"]
            assert len(responses["choices"]) == len(prompts)

            for response in responses["choices"]:
                if "text" in response:
                    api_responses.append(response["text"])
                else:
                    api_responses.append(srsly.json_dumps(response))

        return api_responses
