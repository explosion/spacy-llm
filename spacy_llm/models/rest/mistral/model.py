import warnings
import os
from typing import Iterable, Optional, Any, Dict
from ..base import REST

from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage


class AzureMistral(REST):
    def __init__(
        self,
        name: str,
        endpoint: str,
        config: Dict[Any, Any],
        strict: bool,
        max_tries: int,
        interval: float,
        max_request_time: float,
        context_length: Optional[int],
    ):
        super().__init__(
            name,
            endpoint,
            config,
            strict,
            max_tries,
            interval,
            max_request_time,
            context_length,
        )

    def __call__(self, prompts: Iterable[Iterable[str]]) -> Iterable[Iterable[str]]:
        all_resps = []
        api_key = self._credentials.get("api-key")
        for prompts_doc in prompts:
            doc_resps = []
            for prompt in prompts_doc:
                client = MistralClient(endpoint=self._endpoint, api_key=api_key)

                chat_response = client.chat(
                    model=self._name,
                    messages=[
                        ChatMessage(
                            role="user",
                            content=prompt,
                        )
                    ],
                )
                doc_resps.append(chat_response.choices[0].message.content)
            all_resps.append(doc_resps)
        return all_resps

    @staticmethod
    def _get_context_lengths() -> Dict[str, int]:
        return {
            "azureai": 8192,
        }

    def _verify_auth(self) -> None:
        try:
            self([["test"]])
        except ValueError as err:
            raise err

    @property
    def credentials(self) -> Dict[str, str]:
        # Fetch and check the key
        api_key = os.getenv("MISTRAL_API_KEY")
        if api_key is None:
            warnings.warn(
                "Could not find the API key to access the Mistral AI API. Ensure you have an API key "
                "set up (see "
                ", then make it available as an environment variable 'MISTRAL_API_KEY'."
            )

        # Check the access and get a list of available models to verify the model argument (if not None)
        # Even if the model is None, this call is used as a healthcheck to verify access.
        assert api_key is not None
        return {"api-key": api_key}
