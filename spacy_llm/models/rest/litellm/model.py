import os
import warnings
from typing import Any, Dict, Iterable, List, Optional

import srsly  # type: ignore[import]

from ..base import REST


class LiteLLM(REST):
    """Queries LLMs via the LiteLLM Python SDK, supporting 100+ providers."""

    @property
    def credentials(self) -> Dict[str, str]:
        api_key = os.getenv("LITELLM_API_KEY", "")
        if not api_key:
            warnings.warn(
                "No LITELLM_API_KEY found. LiteLLM will attempt to read provider-specific "
                "API keys from environment variables (e.g. OPENAI_API_KEY, ANTHROPIC_API_KEY)."
            )
        return {"api_key": api_key}

    def _verify_auth(self) -> None:
        pass

    def __call__(self, prompts: Iterable[Iterable[str]]) -> Iterable[Iterable[str]]:
        import litellm

        all_api_responses: List[List[str]] = []

        for prompts_for_doc in prompts:
            api_responses: List[str] = []
            prompts_for_doc = list(prompts_for_doc)

            for prompt in prompts_for_doc:
                kwargs: Dict[str, Any] = {
                    "model": self._name,
                    "messages": [{"role": "user", "content": prompt}],
                    "drop_params": True,
                    **self._config,
                }
                api_key = self._credentials.get("api_key")
                if api_key:
                    kwargs["api_key"] = api_key
                if self._endpoint:
                    kwargs["api_base"] = self._endpoint

                try:
                    response = litellm.completion(**kwargs)
                    api_responses.append(
                        response.choices[0].message.content or ""
                    )
                except Exception as e:
                    if self._strict:
                        raise ValueError(
                            f"Request to LiteLLM API failed: {e}"
                        ) from e
                    else:
                        api_responses.append(srsly.json_dumps({"error": str(e)}))

            all_api_responses.append(api_responses)

        return all_api_responses

    @staticmethod
    def _get_context_lengths() -> Dict[str, int]:
        return {}
