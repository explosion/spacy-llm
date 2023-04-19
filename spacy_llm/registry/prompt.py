from typing import Callable, Iterable

import minichain
import spacy


@spacy.registry.llm("spacy.prompt.MiniChainSimple.v1")
def minichain_simple_prompt() -> Callable[
    [minichain.Backend, Iterable[str]], Iterable[str]
]:
    """Returns Promptable wrapper for Minichain.
    RETURNS (Callable[[minichain.Backend, Iterable[str]], Iterable[str]]:): Callable executing simple prompts on a
        MiniChain backend.
    """

    def prompt(backend: minichain.Backend, prompts: Iterable[str]) -> Iterable[str]:
        @minichain.prompt(backend())
        def _prompt(model: minichain.backend, prompt_text: str) -> str:
            return model(prompt_text)

        return [_prompt(pr).run() for pr in prompts]

    return prompt
