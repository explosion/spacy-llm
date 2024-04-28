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
    TAGS = "http://localhost:11434/api/tags"

class Ollama(REST):
    @property
    def credentials(self) -> Dict[str, str]:
        # No credentials needed for local Ollama server
        return {}
    
    def _verify_auth(self) -> None:
        # Healthcheck: Verify connectivity to Ollama server
        try:
            r = requests.get(Endpoints.TAGS.value, timeout=5)
            r.raise_for_status()
        except (requests.exceptions.RequestException, HTTPError) as ex:
            raise ValueError(
                "Failed to connect to the Ollama server. Please ensure that the server is up and running."
            ) from ex

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
            "llama3": 4096,
            "phi3": 4096,
            "wizardlm2": 4096,
            "mistral": 4096,
            "gemma": 4096,
            "mixtral": 47000,
            "llama2": 4096,
            "codegemma": 4096,
            "command-r": 35000,
            "command-r-plus": 35000,
            "llava": 4096,
            "dbrx": 4096,
            "codellama": 4096,
            "qwen": 4096,
            "dolphin-mixtral": 47000,
            "llama2-uncensored": 4096,
            "mistral-openorca": 4096,
            "deepseek-coder": 4096,
            "phi": 4096,
            "dolphin-mistral": 47000,
            "nomic-embed-text": 4096,
            "nous-hermes2": 4096,
            "orca-mini": 4096,
            "llama2-chinese": 4096,
            "zephyr": 4096,
            "wizard-vicuna-uncensored": 4096,
            "openhermes": 4096,
            "vicuna": 4096,
            "tinyllama": 4096,
            "tinydolphin": 4096,
            "openchat": 4096,
            "starcoder2": 4096,
            "wizardcoder": 4096,
            "stable-code": 4096,
            "starcoder": 4096,
            "neural-chat": 4096,
            "yi": 4096,
            "phind-codellama": 4096,
            "starling-lm": 4096,
            "wizard-math": 4096,
            "falcon": 4096,
            "dolphin-phi": 4096,
            "orca2": 4096,
            "dolphincoder": 4096,
            "mxbai-embed-large": 4096,
            "nous-hermes": 4096,
            "solar": 4096,
            "bakllava": 4096,
            "sqlcoder": 4096,
            "medllama2": 4096,
            "nous-hermes2-mixtral": 47000,
            "wizardlm-uncensored": 4096,
            "dolphin-llama3": 4096,
            "codeup": 4096,
            "stablelm2": 4096,
            "everythinglm": 16384,
            "all-minilm": 4096,
            "samantha-mistral": 4096,
            "yarn-mistral": 128000,
            "stable-beluga": 4096,
            "meditron": 4096,
            "yarn-llama2": 128000,
            "deepseek-llm": 4096,
            "llama-pro": 4096,
            "magicoder": 4096,
            "stablelm-zephyr": 4096,
            "codebooga": 4096,
            "codeqwen": 4096,
            "mistrallite": 8192,
            "wizard-vicuna": 4096,
            "nexusraven": 4096,
            "xwinlm": 4096,
            "goliath": 4096,
            "open-orca-platypus2": 4096,
            "wizardlm": 4096,
            "notux": 4096,
            "megadolphin": 4096,
            "duckdb-nsql": 4096,
            "alfred": 4096,
            "notus": 4096,
            "snowflake-arctic-embed": 4096
        }
