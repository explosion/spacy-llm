import os
import warnings
from enum import Enum
from typing import Any, Dict, Iterable, List, Sized

import requests  # type: ignore[import]
import srsly  # type: ignore[import]
from requests import HTTPError

from ..base import REST


class Endpoints(str, Enum):
    GENERATE = "http://localhost:11434/api/generate"
    EMBEDDINGS = "http://localhost:11434/api/embeddings"

class Ollama(REST):
    @property
    def credentials(self) -> Dict[str, str]:
        # No credentials needed for local Ollama server
        return {}
    
    def _verify_auth(self) -> None:
        # TODO: Verify connectivity to Ollama server
        pass

    def __call__(self, prompts: Iterable[Iterable[str]]) -> Iterable[Iterable[str]]:
        headers = {
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
                    json={**json_data, **self._config, "model": self._name, "stream": False},
                    timeout=self._max_request_time,
                )
                try:
                    r.raise_for_status()
                except HTTPError as ex:
                    res_content = r.text
                    # Include specific error message in exception.
                    raise ValueError(
                        f"Request to Ollama API failed: {res_content}"
                    ) from ex
                
                response = r.json()

                if "error" in response:
                    if self._strict:
                        raise ValueError(f"API call failed: {response['error']}.")
                    else:
                        assert isinstance(prompts_for_doc, Sized)
                        return {"error": [response['error']] * len(prompts_for_doc)}

                return response

            for prompt in prompts_for_doc:
                responses = _request({"prompt": prompt})
                if "error" in responses:
                    return responses["error"]

                api_responses.append(responses["response"])

            all_api_responses.append(api_responses)

        return all_api_responses

    @staticmethod
    def _get_context_lengths() -> Dict[str, int]:
        return {
            "mistral": 4096
        }
