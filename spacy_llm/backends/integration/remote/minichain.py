from typing import Any, Callable, Dict, Iterable, Optional

from spacy.util import SimpleFrozenDict

from ....compat import has_minichain, minichain
from ....registry import registry
from .base import RemoteBackend


def _check_installation() -> None:
    """Checks whether `minichain` is installed. Raises an error otherwise."""
    if not has_minichain:
        raise ValueError(
            "The MiniChain backend requires `minichain` to be installed, which it is not. See "
            "https://github.com/srush/MiniChain for installation instructions."
        )


@registry.llm_queries("spacy.RunMiniChain.v1")
def query_minichain() -> (
    Callable[["minichain.backend.Backend", Iterable[str]], Iterable[str]]
):
    """Returns query Callable for MiniChain.
    RETURNS (Callable[["minichain.backend.Backend", Iterable[str]], Iterable[str]]): Callable executing simple prompts
        on the specified MiniChain backend.
    """

    def prompt(
        backend: "minichain.backend.Backend", prompts: Iterable[str]
    ) -> Iterable[str]:
        @minichain.prompt(backend)
        def _prompt(model: "minichain.base.Prompt.Model", prompt_text: str) -> str:
            return model(prompt_text)

        return [_prompt(pr).run() for pr in prompts]

    return prompt


@registry.llm_backends("spacy.MiniChain.v1")
def backend_minichain(
    api: str,
    query: Optional[
        Callable[["minichain.backend.Backend", Iterable[str]], Iterable[str]]
    ] = None,
    config: Dict[Any, Any] = SimpleFrozenDict(),
) -> Callable[[Iterable[str]], Iterable[str]]:
    """Returns Callable using MiniChain backend to prompt specified API.
    api (str): Name of any API/class in minichain.backend, e. g. "OpenAI".
    query (Optional[Callable[["minichain.backend.Backend", Iterable[str]], Iterable[str]]]): Callable implementing querying this
        API.
    config (Dict[Any, Any]): LLM config arguments passed on to the initialization of the minichain.Backend
        instance.
    RETURNS (Callable[[Iterable[str]], Iterable[str]]]): Callable querying the specified API using the
        specified backend.
    """
    _check_installation()

    if "model" not in config:
        raise ValueError("The LLM model must be specified in the config.")

    if hasattr(minichain.backend, api):
        return RemoteBackend(
            integration=getattr(minichain.backend, api)(**config),
            query=query_minichain() if query is None else query,
        )
    else:
        raise KeyError(
            f"The requested API {api} is not available in `minichain.backend`."
        )
