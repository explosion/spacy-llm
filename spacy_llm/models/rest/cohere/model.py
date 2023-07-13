import os
import warnings
from enum import Enum
from typing import Any, Dict, Iterable, List, Sized, Tuple

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
            self(["test"])
        except ValueError as err:
            if "invalid api token" in str(err):
                warnings.warn(
                    "Authentication with provided API key failed. Please double-check you provided the correct "
                    "credentials."
                )
            else:
                raise err

    def __call__(self, prompts: Iterable[str]) -> Iterable[str]:
        headers = {
            **self._credentials,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        api_responses: List[str] = []
        prompts = list(prompts)

        def _request(json_data: Dict[str, Any]) -> Dict[str, Any]:
            r = self.retry(
                call_method=requests.post,
                url=self._endpoint,
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
                    f"Request to Cohere API failed: {res_content.get('message', {})}"
                ) from ex
            response = r.json()

            # Cohere returns a 'message' key when there is an error
            # in the response.
            if "message" in response:
                if self._strict:
                    raise ValueError(f"API call failed: {response}.")
                else:
                    assert isinstance(prompts, Sized)
                    return {"error": [srsly.json_dumps(response)] * len(prompts)}
            return response

        # Cohere API currently doesn't accept batch prompts, so we're making
        # a request for each iteration. This approach can be prone to rate limit
        # errors. In practice, you can adjust _max_request_time so that the
        # timeout is larger.
        responses = [_request({"prompt": prompt}) for prompt in prompts]
        for response in responses:
            for result in response["generations"]:
                if "text" in result:
                    # Although you can set the number of completions in Cohere
                    # to be greater than 1, we only need to return a single value.
                    # In this case, we will just return the very first output.
                    api_responses.append(result["text"])
                    break
                else:
                    api_responses.append(srsly.json_dumps(response))

        return api_responses

    @classmethod
    def get_model_names(cls) -> Tuple[str, ...]:
        return "command", "command-light", "command-light-nightly", "command-nightly"
