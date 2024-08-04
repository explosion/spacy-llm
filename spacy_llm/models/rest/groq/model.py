import os
import warnings
from enum import Enum
from typing import Any, Dict, Iterable, List, Sized

import requests  # type: ignore[import]
import srsly  # type: ignore[import]
from requests import HTTPError

from ..base import REST


class Endpoints(str, Enum):
    CHAT = "https://api.groq.com/openai/v1/chat/completions"
    NON_CHAT = CHAT # Completion endpoints are not available 


class Groq(REST):
    @property
    def credentials(self) -> Dict[str, str]:
        # Fetch and check the key
        api_key = os.getenv("GROQ_API_KEY")
        # api_org = os.getenv("OPENAI_API_ORG")
        if api_key is None:
            warnings.warn(
                "Could not find the API key to access the OpenAI API. Ensure you have an API key "
                "set up via https://console.groq.com/keys, then make it available as "
                "an environment variable 'GROQ_API_KEY'."
            )

        # Check the access and get a list of available models to verify the model argument (if not None)
        # Even if the model is None, this call is used as a healthcheck to verify access.
        headers = {
            "Authorization": f"Bearer {api_key}",
        }
        # if api_org:
        #     headers["OpenAI-Organization"] = api_org

        return headers

    def _verify_auth(self) -> None:
        r = self.retry(
            call_method=requests.get,
            url="https://api.groq.com/openai/v1/models",
            headers=self._credentials,
            timeout=self._max_request_time,
        )
        if r.status_code == 422:
            warnings.warn(
                "Could not access api.groq.com -- 422 permission denied."
                "Visit https://console.groq.com/keys to check your API keys."
            )
        elif r.status_code != 200:
            if "Incorrect API key" in r.text:
                warnings.warn(
                    "Authentication with provided API key failed. Please double-check you provided the correct "
                    "credentials."
                )
            else:
                warnings.warn(
                    f"Error accessing api.groq.com({r.status_code}): {r.text}"
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
                        f"Request to Groq API failed: {res_content.get('error', {}).get('message', str(res_content))}"
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

            # The Groq API doesn't support NON_CHAT (yet), so we have to send individual requests.

            if self._endpoint == Endpoints.NON_CHAT:
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
            "gemma2-9b-it": 8192,
            "gemma-7b-it": 8192,
            "llama-3.1-70b-versatile": 131072,
            "llama-3.1-8b-instant": 131072,
            "llama3-70b-8192": 8192,
            "llama3-8b-8192": 8192,
            "llama3-groq-70b-8192-tool-use-preview": 8192,
            "llama3-groq-8b-8192-tool-use-preview": 8192,
            "llama-guard-3-8b": 8192,
            "mixtral-8x7b-32768": 32768,
            "whisper-large-v3": 1500
        }

