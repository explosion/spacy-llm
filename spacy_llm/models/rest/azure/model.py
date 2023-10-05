import os
import warnings
from enum import Enum
from typing import Any, Dict, Iterable, List, Sized, Tuple

import requests  # type: ignore[import]
import srsly  # type: ignore[import]
from requests import HTTPError

from ..base import REST


class ModelType(str, Enum):
    COMPLETION = "completions"
    CHAT = "chat"


class AzureOpenAI(REST):
    def __init__(
        self,
        name: str,
        endpoint: str,
        config: Dict[Any, Any],
        strict: bool,
        max_tries: int,
        interval: float,
        max_request_time: float,
        model_type: ModelType,
        api_version: str = "2023-05-15",
    ):
        self._model_type = model_type
        self._api_version = api_version
        super().__init__(
            name=name,
            endpoint=endpoint,
            config=config,
            strict=strict,
            max_tries=max_tries,
            interval=interval,
            max_request_time=max_request_time,
        )

    @property
    def endpoint(self) -> str:
        """Returns fully formed endpoint URL.
        RETURNS (str): Fully formed endpoint URL.
        """
        return (
            self._endpoint
            + ("" if self._endpoint.endswith("/") else "/")
            + f"openai/deployments/{self._name}/{self._model_type.value}"
        )

    @property
    def credentials(self) -> Dict[str, str]:
        # Fetch and check the key
        api_key = os.getenv("AZURE_OPENAI_KEY")
        if api_key is None:
            warnings.warn(
                "Could not find the API key to access the Azure OpenAI API. Ensure you have an API key "
                "set up (see "
                "https://learn.microsoft.com/en-us/azure/ai-services/openai/quickstart?pivots=rest-api&tabs=bash#set-up"
                ", then make it available as an environment variable 'AZURE_OPENAI_KEY'."
            )

        # Check the access and get a list of available models to verify the model argument (if not None)
        # Even if the model is None, this call is used as a healthcheck to verify access.
        assert api_key is not None
        return {"api-key": api_key}

    def _verify_auth(self) -> None:
        try:
            self(["test"])
        except ValueError as err:
            raise err

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
                url=self.endpoint,
                headers=headers,
                json={**json_data, **self._config},
                timeout=self._max_request_time,
                params={"api-version": self._api_version},
            )
            try:
                r.raise_for_status()
            except HTTPError as ex:
                res_content = srsly.json_loads(r.content.decode("utf-8"))
                # Include specific error message in exception.
                raise ValueError(
                    f"Request to Azure OpenAI API failed: "
                    f"{res_content.get('error', {}).get('message', str(res_content))}"
                ) from ex
            responses = r.json()

            # todo check if this is the same
            if "error" in responses:
                if self._strict:
                    raise ValueError(f"API call failed: {responses}.")
                else:
                    assert isinstance(prompts, Sized)
                    return {"error": [srsly.json_dumps(responses)] * len(prompts)}

            return responses

        # The (Azure) OpenAI API doesn't support batching yet, so we have to send individual requests.
        # https://learn.microsoft.com/en-us/answers/questions/1334800/batching-requests-in-azure-openai

        if self._model_type == ModelType.CHAT:
            # Note: this is yet (2023-10-05) untested, as Azure doesn't seem to allow the deployment of a chat model
            # yet.
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

        elif self._model_type == ModelType.COMPLETION:
            for prompt in prompts:
                responses = _request({"prompt": prompt})
                if "error" in responses:
                    return responses["error"]

                # Process responses.
                assert len(responses["choices"]) == 1
                response = responses["choices"][0]
                api_responses.append(response.get("text", srsly.json_dumps(response)))

        return api_responses

    @classmethod
    def get_model_names(cls) -> Tuple[str, ...]:
        # We treat the deployment name as "model name", hence it can be arbitrary.
        return ("",)

    def _check_model(self) -> None:
        # We treat the deployment name as "model name", hence it can be arbitrary.
        pass
