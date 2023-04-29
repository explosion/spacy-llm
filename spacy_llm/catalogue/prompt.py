from typing import Callable, Iterable

from ..util import registry
from ..compat import langchain, minichain


@registry.prompts("spacy-llm.MiniChainSimple.v1")
def minichain_simple_prompt() -> Callable[
    ["minichain.Backend", Iterable[str]], Iterable[str]
]:
    """Returns prompt Callable for MiniChain.
    RETURNS (Callable[[minichain.Backend, Iterable[str]], Iterable[str]]:): Callable executing simple prompts on a
        MiniChain backend.
    """

    def prompt(backend: "minichain.Backend", prompts: Iterable[str]) -> Iterable[str]:
        @minichain.prompt(backend)
        def _prompt(model: "minichain.backend", prompt_text: str) -> str:
            return model(prompt_text)

        return [_prompt(pr).run() for pr in prompts]

    return prompt


@registry.prompts("spacy-llm.LangChainSimple.v1")
def langchain_simple_prompt() -> Callable[
    ["langchain.llms.BaseLLM", Iterable[str]], Iterable[str]
]:
    """Returns prompt Callable for LangChain.
    RETURNS (Callable[[langchain.llms.BaseLLM, Iterable[str]], Iterable[str]]:): Callable executing simple prompts on a
        LangChain backend.
    """

    def prompt(
        backend: "langchain.llms.BaseLLM", prompts: Iterable[str]
    ) -> Iterable[str]:
        return [backend(pr) for pr in prompts]

    return prompt
