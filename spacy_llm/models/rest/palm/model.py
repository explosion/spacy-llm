import os
import warnings
from enum import Enum
from typing import Any, Dict, Iterable, List, Sized

import requests  # type: ignore[import]
import srsly  # type: ignore[import]
from requests import HTTPError

from ..base import REST


class Endpoints(str, Enum):
    TEXT = "https://generativelanguage.googleapis.com/v1beta3/models/{model}:generateText?key={api_key}"
    MSG = "https://generativelanguage.googleapis.com/v1beta3/models/{model}:generateMessage?key={api_key}"


class PaLM(REST):
    @property
    def credentials(self) -> Dict[str, str]:
        api_key = os.getenv("PALM_API_KEY")
        if api_key is None:
            warnings.warn(
                "Could not find the API key to access the Cohere API. Ensure you have an API key "
                "set up via https://cloud.google.com/docs/authentication/api-keys#rest, then make it available as "
                "an environment variable 'PALM_API_KEY'."
            )

        assert api_key is not None
        return {"api_key": api_key}

    def _verify_auth(self) -> None:
        try:
            self([["What's 2+2?"]])
        except ValueError as err:
            if "API key not valid" in str(err):
                warnings.warn(
                    "Authentication with provided API key failed. Please double-check you provided the correct "
                    "credentials."
                )
            else:
                raise err

    def __call__(self, prompts: Iterable[Iterable[str]]) -> Iterable[Iterable[str]]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        url = self._endpoint.format(
            model=self._name, api_key=self._credentials["api_key"]
        )
        all_api_responses: List[List[str]] = []

        for prompts_for_doc in prompts:
            api_responses: List[str] = []
            prompts_for_doc = list(prompts_for_doc)

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
                    error_message = res_content.get("error", {}).get("message", {})
                    # Catching other types of HTTPErrors (e.g., "429: too many requests")
                    raise ValueError(
                        f"Request to PaLM API failed: {error_message}"
                    ) from ex
                response = r.json()

                # PaLM returns a 'filter' key when a message was filtered due to safety concerns.
                if "filters" in response:
                    if self._strict:
                        raise ValueError(f"API call failed: {response}.")
                    else:
                        assert isinstance(prompts_for_doc, Sized)
                        return {
                            "error": [srsly.json_dumps(response)] * len(prompts_for_doc)
                        }

                return response

            # PaLM API currently doesn't accept batch prompts, so we're making
            # a request for each iteration. This approach can be prone to rate limit
            # errors. In practice, you can adjust _max_request_time so that the
            # timeout is larger.
            uses_chat = "chat" in self._name
            responses = [
                _request(
                    {
                        "prompt": {"text": prompt}
                        if not uses_chat
                        else {"messages": [{"content": prompt}]}
                    }
                )
                for prompt in prompts_for_doc
            ]
            for response in responses:
                if "candidates" in response:
                    # Although you can set the number of candidates in PaLM to be greater than 1, we only need to return a
                    # single value. In this case, we will just return the very first output.
                    api_responses.append(
                        response["candidates"][0].get(
                            "content" if uses_chat else "output",
                            srsly.json_dumps(response),
                        )
                    )
                else:
                    api_responses.append(srsly.json_dumps(response))

            all_api_responses.append(api_responses)

        return all_api_responses

    @staticmethod
    def _get_context_lengths() -> Dict[str, int]:
        return {
            "text-bison-001": 8192,
            "chat-bison-001": 8192,
        }
