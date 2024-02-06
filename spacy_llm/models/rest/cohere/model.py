import os
import warnings
from enum import Enum
from typing import Any, Dict, Iterable, List, Sized

import requests  # type: ignore[import]
import srsly  # type: ignore[import]
from requests import HTTPError

from ..base import REST


class Endpoints(str, Enum):
    COMPLETION = "https://api.cohere.ai/v1/generate"


class Cohere(REST):
    @property
    def credentials(self) -> Dict[str, str]:
        api_key = os.getenv("CO_API_KEY")
        if api_key is None:
            warnings.warn(
                "Could not find the API key to access the Cohere API. Ensure you have an API key "
                "set up via https://dashboard.cohere.ai/api-keys, then make it available as "
                "an environment variable 'CO_API_KEY'."
            )

        return {"Authorization": f"Bearer {api_key}"}

    def _verify_auth(self) -> None:
        try:
            self([["test"]])
        except ValueError as err:
            if "invalid api token" in str(err):
                warnings.warn(
                    "Authentication with provided API key failed. Please double-check you provided the correct "
                    "credentials."
                )
            else:
                raise err

    def __call__(self, prompts: Iterable[Iterable[str]]) -> Iterable[Iterable[str]]:
        headers = {
            **self._credentials,
            "Content-Type": "application/json",
            "Accept": "application/json",
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
                error_message = res_content.get("message", {})
                # Catch 'blocked output' and 'blocked input' errors from Cohere
                # This usually happens when it detects violations in their Usage guidelines.
                # Unfortunately Cohere returns this as an HTTPError, so it cannot be caught in the response.
                if "blocked" in error_message:
                    # Only raise an error when strict. If strict is False, do
                    # nothing and parse the response as usual.
                    if self._strict:
                        raise ValueError(
                            f"Cohere API returned a blocking error. {error_message}. "
                            "If you wish to ignore and continue, you can pass 'False' to the 'strict' argument of this model. "
                            "However, note that this will affect how spacy-llm parses the response."
                        ) from ex
                else:
                    # Catching other types of HTTPErrors (e.g., "429: too many requests")
                    raise ValueError(
                        f"Request to Cohere API failed: {error_message}"
                    ) from ex
            response = r.json()

            # Cohere returns a 'message' key when there is an error
            # in the response.
            if "message" in response:
                if self._strict:
                    raise ValueError(f"API call failed: {response}.")
                else:
                    assert isinstance(prompts_for_doc, Sized)
                    return {
                        "error": [srsly.json_dumps(response)] * len(prompts_for_doc)
                    }

            return response

        # Cohere API currently doesn't accept batch prompts, so we're making
        # a request for each iteration. This approach can be prone to rate limit
        # errors. In practice, you can adjust _max_request_time so that the
        # timeout is larger.
        responses = [_request({"prompt": prompt}) for prompt in prompts_for_doc]
        for response in responses:
            if "generations" in response:
                for result in response["generations"]:
                    if "text" in result:
                        # Although you can set the number of completions in Cohere
                        # to be greater than 1, we only need to return a single value.
                        # In this case, we will just return the very first output.
                        api_responses.append(result["text"])
                        break
                    else:
                        api_responses.append(srsly.json_dumps(response))
            else:
                api_responses.append(srsly.json_dumps(response))

        all_api_responses.append(api_responses)

        return all_api_responses

    @staticmethod
    def _get_context_lengths() -> Dict[str, int]:
        return {
            "command": 4096,
            "command-light": 4096,
            "command-light-nightly": 4096,
            "command-nightly": 4096,
        }
