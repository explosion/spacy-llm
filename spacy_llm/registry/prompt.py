from typing import Callable, Iterable

from ..compat import langchain, minichain
from .util import registry


@registry.prompts("spacy-llm.MiniChainSimple.v1")
def minichain_simple_prompt(
    api: Callable[[], "minichain.Backend"]
) -> Callable[[Iterable[str]], Iterable[str]]:
    """Returns prompt Callable for MiniChain.
    api (Callable[[], "minichain.Backend"]): Callable generating a minichain.Backend instance.
    RETURNS (Callable[[Iterable[str]], Iterable[str]]): Callable executing simple prompts on the specified MiniChain
        backend.
    """

    def prompt(backend: "minichain.Backend", prompts: Iterable[str]) -> Iterable[str]:
        @minichain.prompt(backend)
        def _prompt(model: "minichain.backend", prompt_text: str) -> str:
            return model(prompt_text)

        return [_prompt(pr).run() for pr in prompts]

    _api = api()
    return lambda prompts: prompt(_api, prompts)


@registry.prompts("spacy-llm.LangChainSimple.v1")
def langchain_simple_prompt(
    api: Callable[[], "langchain.llms.BaseLLM"]
) -> Callable[[Iterable[str]], Iterable[str]]:
    """Returns prompt Callable for LangChain.
    RETURNS (Callable[[Iterable[str]], Iterable[str]]:): Callable executing simple prompts on the specified LangChain
        backend.
    """

    def prompt(
        backend: "langchain.llms.BaseLLM", prompts: Iterable[str]
    ) -> Iterable[str]:
        return [backend(pr) for pr in prompts]

    _api = api()
    return lambda prompts: prompt(_api, prompts)
