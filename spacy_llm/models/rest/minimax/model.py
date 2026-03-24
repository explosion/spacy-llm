import os
import re
import warnings
from enum import Enum
from typing import Any, Dict, Iterable, List, Sized

import requests  # type: ignore[import]
import srsly  # type: ignore[import]
from requests import HTTPError

from ..base import REST


class Endpoints(str, Enum):
    CHAT = "https://api.minimax.io/v1/chat/completions"


class MiniMax(REST):
    @property
    def credentials(self) -> Dict[str, str]:
        api_key = os.getenv("MINIMAX_API_KEY")
        if api_key is None:
            warnings.warn(
                "Could not find the API key to access the MiniMax API. Ensure you have an API key "
                "set up via https://platform.minimaxi.com/, then make it available as "
                "an environment variable 'MINIMAX_API_KEY'."
            )

        return {
            "Authorization": f"Bearer {api_key}",
        }

    def _verify_auth(self) -> None:
        # MiniMax doesn't have a dedicated /v1/models endpoint, so we verify
        # auth by making a minimal test request.
        try:
            self([["test"]])
        except ValueError as err:
            if "authentication" in str(err).lower() or "api key" in str(err).lower():
                warnings.warn(
                    "Authentication with provided API key failed. Please double-check you provided "
                    "the correct credentials."
                )
            else:
                raise err

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
                    raise ValueError(
                        f"Request to MiniMax API failed: "
                        f"{res_content.get('error', {}).get('message', str(res_content))}"
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

            # MiniMax uses an OpenAI-compatible chat completions API,
            # so we send individual requests per prompt (no batching).
            for prompt in prompts_for_doc:
                responses = _request(
                    {"messages": [{"role": "user", "content": prompt}]}
                )
                if "error" in responses:
                    return responses["error"]

                assert len(responses["choices"]) == 1
                response = responses["choices"][0]
                content = response.get("message", {}).get(
                    "content", srsly.json_dumps(response)
                )
                # Strip <think>...</think> tags from thinking models.
                content = re.sub(
                    r"<think>.*?</think>\s*", "", content, flags=re.DOTALL
                )
                api_responses.append(content)

            all_api_responses.append(api_responses)

        return all_api_responses

    @staticmethod
    def _get_context_lengths() -> Dict[str, int]:
        return {
            "MiniMax-M2.7": 1048576,
            "MiniMax-M2.7-highspeed": 1048576,
            "MiniMax-M2.5": 204800,
            "MiniMax-M2.5-highspeed": 204800,
        }
