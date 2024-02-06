import os
import warnings
from enum import Enum
from typing import Any, Dict, Iterable, List, Sized

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
                "an environment variable 'OPENAI_API_KEY'."
            )

        # Check the access and get a list of available models to verify the model argument (if not None)
        # Even if the model is None, this call is used as a healthcheck to verify access.
        headers = {
            "Authorization": f"Bearer {api_key}",
        }
        if api_org:
            headers["OpenAI-Organization"] = api_org

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

    def __call__(self, prompts: Iterable[Iterable[str]]) -> Iterable[Iterable[str]]:
        headers = {
            **self._credentials,
            "Content-Type": "application/json",
        }
        all_api_responses: List[List[str]] = []

        for prompts_for_doc in prompts:
            api_responses: List[str] = []
            prompts_for_doc = list(prompts_for_doc)

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
                        assert isinstance(prompts_for_doc, Sized)
                        return {
                            "error": [srsly.json_dumps(responses)]
                            * len(prompts_for_doc)
                        }

                return responses

            # The OpenAI API doesn't support batching for /chat/completions yet, so we have to send individual requests.

            if self._endpoint == Endpoints.NON_CHAT:
                responses = _request({"prompt": prompts_for_doc})
                if "error" in responses:
                    return responses["error"]
                assert len(responses["choices"]) == len(prompts_for_doc)

                for response in responses["choices"]:
                    if "text" in response:
                        api_responses.append(response["text"])
                    else:
                        api_responses.append(srsly.json_dumps(response))

            else:
                for prompt in prompts_for_doc:
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

            all_api_responses.append(api_responses)

        return all_api_responses

    @staticmethod
    def _get_context_lengths() -> Dict[str, int]:
        return {
            # gpt-4
            "gpt-4": 8192,
            "gpt-4-0314": 8192,
            "gpt-4-32k": 32768,
            "gpt-4-32k-0314": 32768,
            # gpt-3.5
            "gpt-3.5-turbo": 4097,
            "gpt-3.5-turbo-16k": 16385,
            "gpt-3.5-turbo-0613": 4097,
            "gpt-3.5-turbo-0613-16k": 16385,
            "gpt-3.5-turbo-instruct": 4097,
            # text-davinci
            "text-davinci-002": 4097,
            "text-davinci-003": 4097,
            # others
            "code-davinci-002": 8001,
            "text-curie-001": 2049,
            "text-babbage-001": 2049,
            "text-ada-001": 2049,
            "davinci": 2049,
            "curie": 2049,
            "babbage": 2049,
            "ada": 2049,
        }
