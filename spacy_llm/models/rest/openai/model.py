import os
import warnings
from enum import Enum
from typing import Any, Dict, Iterable, List, Sized, Tuple

import requests  # type: ignore[import]
import srsly  # type: ignore[import]
from requests import HTTPError

from ..base import REST


class Endpoints(str, Enum):
    CHAT = "https://api.openai.com/v1/chat/completions"
    NON_CHAT = "https://api.openai.com/v1/completions"


class OpenAI(REST):
    @property
    def credentials(self) -> Dict[str, str]:
        # Fetch and check the key
        api_key = os.getenv("OPENAI_API_KEY")
        api_org = os.getenv("OPENAI_API_ORG")
        if api_key is None:
            warnings.warn(
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

        # Ensure endpoint is supported.
        if self._endpoint not in (Endpoints.NON_CHAT, Endpoints.CHAT):
            raise ValueError(
                f"Endpoint {self._endpoint} isn't supported. Please use one of: {Endpoints.CHAT}, {Endpoints.NON_CHAT}."
            )

        return headers

    def _verify_auth(self) -> None:
        r = self.retry(
            call_method=requests.get,
            url="https://api.openai.com/v1/models",
            headers=self._credentials,
            timeout=self._max_request_time,
        )
        if r.status_code == 422:
            warnings.warn(
                "Could not access api.openai.com -- 422 permission denied."
                "Visit https://platform.openai.com/account/api-keys to check your API keys."
            )
        elif r.status_code != 200:
            if "Incorrect API key" in r.text:
                warnings.warn(
                    "Authentication with provided API key failed. Please double-check you provided the correct "
                    "credentials."
                )
            else:
                warnings.warn(
                    f"Error accessing api.openai.com ({r.status_code}): {r.text}"
                )

        response = r.json()["data"]
        models = [response[i]["id"] for i in range(len(response))]
        if self._name not in models:
            raise ValueError(
                f"The specified model '{self._name}' is not available. Choices are: {sorted(set(models))}"
            )

    def __call__(self, prompts: Iterable[str]) -> Iterable[str]:
        headers = {
            **self._credentials,
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
                raise ValueError(
                    f"Request to OpenAI API failed: {res_content.get('error', {}).get('message', str(res_content))}"
                ) from ex
            responses = r.json()

            if "error" in responses:
                if self._strict:
                    raise ValueError(f"API call failed: {responses}.")
                else:
                    assert isinstance(prompts, Sized)
                    return {"error": [srsly.json_dumps(responses)] * len(prompts)}

            return responses

        if self._endpoint == Endpoints.CHAT:
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

        elif self._endpoint == Endpoints.NON_CHAT:
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

    @classmethod
    def get_model_names(cls) -> Tuple[str, ...]:
        return (
            # gpt-4
            "gpt-4",
            "gpt-4-0314",
            "gpt-4-32k",
            "gpt-4-32k-0314",
            # gpt-3.5
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k",
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-0613-16k",
            # text-davinci
            "text-davinci-002",
            "text-davinci-003",
            # others
            "code-davinci-002",
            "text-curie-001",
            "text-babbage-001",
            "text-ada-001",
            "davinci",
            "curie",
            "babbage",
            "ada",
        )
