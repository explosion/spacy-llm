import os
from typing import Iterable, Dict, Any, List

import requests  # type: ignore
import srsly


class Backend:
    """Queries LLMs via their REST APIs."""

    _SUPPORTED_APIS = ("OpenAI",)

    def __init__(self, api: str, config: Dict[Any, Any], strict: bool = True):
        """Initializes new Backend instance.
        api (str): Name of LLM API.
        config (Dict[Any, Any]): Config passed on to LLM API.
        strict (bool): If True, ValueError is raised if the LLM API returns a malformed response (i. e. any kind of JSON
            or other response object that does not conform to the expectation of how a well-formed response object from
            this API should look like). If False, the API error responses are returned by __call__(), but no error will
            be raised.
            Note that only response object structure will be checked, not the prompt response text per se.
        """
        if api not in Backend._SUPPORTED_APIS:
            raise ValueError(
                f"{api} is not one of the supported APIs ({Backend._SUPPORTED_APIS})."
            )
        self._api = api
        self._config = config
        self._strict = strict
        self._calls = {"OpenAI": self._call_openai}

    def __call__(self, prompts: Iterable[str]) -> Iterable[str]:
        """Executes prompts on specified API.
        prompts (Iterable[str]): Prompts to execute.
        RETURNS (Iterable[str]): API responses.
        """
        return self._calls[self._api](prompts)

    def _call_openai(self, prompts: Iterable[str]) -> Iterable[str]:
        """
        Calls OpenAI API. Note that this currently only implements the completions endpoint
        (https://api.openai.com/v1/completions), not the chat endpoint (https://api.openai.com/v1/chat/completions).
        See https://platform.openai.com/docs/api-reference/.
        prompts (Iterable[str]): Prompts to execute.
        RETURNS (Iterable[str]): API responses.
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f'Bearer {os.getenv("OPENAI_API_KEY", "")}',
        }
        api_responses: List[str] = []

        for prompt in prompts:
            json_data = {"prompt": prompt, **self._config}
            response = requests.post(
                "https://api.openai.com/v1/completions", headers=headers, json=json_data
            ).json()
            if "error" in response:
                if self._strict:
                    raise ValueError(f"API call failed: {response}.")
                else:
                    api_responses.append(srsly.json_dumps(response))
            else:
                api_responses.append(response["choices"][0]["text"])

        return api_responses
