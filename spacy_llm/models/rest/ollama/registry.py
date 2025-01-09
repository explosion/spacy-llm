from typing import Any, Dict

from confection import SimpleFrozenDict

from ....registry import registry
from .model import Endpoints, Ollama


@registry.llm_models("spacy.Ollama.v1")
def ollama_llama3(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "llama3",
    strict: bool = Ollama.DEFAULT_STRICT,
    max_tries: int = Ollama.DEFAULT_MAX_TRIES,
    interval: float = Ollama.DEFAULT_INTERVAL,
    max_request_time: float = Ollama.DEFAULT_MAX_REQUEST_TIME,
    context_length: int = 4096,
) -> Ollama:
    """Returns Ollama instance for 'llama3' model."""
    return Ollama(
        name=name,
        endpoint=Endpoints.GENERATE.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Ollama.v1")
def ollama_phi3(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "phi3",
    strict: bool = Ollama.DEFAULT_STRICT,
    max_tries: int = Ollama.DEFAULT_MAX_TRIES,
    interval: float = Ollama.DEFAULT_INTERVAL,
    max_request_time: float = Ollama.DEFAULT_MAX_REQUEST_TIME,
    context_length: int = 4096,
) -> Ollama:
    """Returns Ollama instance for 'phi3' model."""
    return Ollama(
        name=name,
        endpoint=Endpoints.GENERATE.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Ollama.v1")
def ollama_wizardlm2(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "wizardlm2",
    strict: bool = Ollama.DEFAULT_STRICT,
    max_tries: int = Ollama.DEFAULT_MAX_TRIES,
    interval: float = Ollama.DEFAULT_INTERVAL,
    max_request_time: float = Ollama.DEFAULT_MAX_REQUEST_TIME,
    context_length: int = 4096,
) -> Ollama:
    """Returns Ollama instance for 'wizardlm2' model."""
    return Ollama(
        name=name,
        endpoint=Endpoints.GENERATE.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Ollama.v1")
def ollama_mistral(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "mistral",
    strict: bool = Ollama.DEFAULT_STRICT,
    max_tries: int = Ollama.DEFAULT_MAX_TRIES,
    interval: float = Ollama.DEFAULT_INTERVAL,
    max_request_time: float = Ollama.DEFAULT_MAX_REQUEST_TIME,
    context_length: int = 4096,
) -> Ollama:
    """Returns Ollama instance for 'mistral' model."""
    return Ollama(
        name=name,
        endpoint=Endpoints.GENERATE.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Ollama.v1")
def ollama_gemma(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "gemma",
    strict: bool = Ollama.DEFAULT_STRICT,
    max_tries: int = Ollama.DEFAULT_MAX_TRIES,
    interval: float = Ollama.DEFAULT_INTERVAL,
    max_request_time: float = Ollama.DEFAULT_MAX_REQUEST_TIME,
    context_length: int = 4096,
) -> Ollama:
    """Returns Ollama instance for 'gemma' model."""
    return Ollama(
        name=name,
        endpoint=Endpoints.GENERATE.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Ollama.v1")
def ollama_mixtral(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "mixtral",
    strict: bool = Ollama.DEFAULT_STRICT,
    max_tries: int = Ollama.DEFAULT_MAX_TRIES,
    interval: float = Ollama.DEFAULT_INTERVAL,
    max_request_time: float = Ollama.DEFAULT_MAX_REQUEST_TIME,
    context_length: int = 47000,
) -> Ollama:
    """Returns Ollama instance for 'mixtral' model."""
    return Ollama(
        name=name,
        endpoint=Endpoints.GENERATE.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Ollama.v1")
def ollama_llama2(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "llama2",
    strict: bool = Ollama.DEFAULT_STRICT,
    max_tries: int = Ollama.DEFAULT_MAX_TRIES,
    interval: float = Ollama.DEFAULT_INTERVAL,
    max_request_time: float = Ollama.DEFAULT_MAX_REQUEST_TIME,
    context_length: int = 4096,
) -> Ollama:
    """Returns Ollama instance for 'llama2' model."""
    return Ollama(
        name=name,
        endpoint=Endpoints.GENERATE.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Ollama.v1")
def ollama_codegemma(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "codegemma",
    strict: bool = Ollama.DEFAULT_STRICT,
    max_tries: int = Ollama.DEFAULT_MAX_TRIES,
    interval: float = Ollama.DEFAULT_INTERVAL,
    max_request_time: float = Ollama.DEFAULT_MAX_REQUEST_TIME,
    context_length: int = 4096,
) -> Ollama:
    """Returns Ollama instance for 'codegemma' model."""
    return Ollama(
        name=name,
        endpoint=Endpoints.GENERATE.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Ollama.v1")
def ollama_command_r(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "command-r",
    strict: bool = Ollama.DEFAULT_STRICT,
    max_tries: int = Ollama.DEFAULT_MAX_TRIES,
    interval: float = Ollama.DEFAULT_INTERVAL,
    max_request_time: float = Ollama.DEFAULT_MAX_REQUEST_TIME,
    context_length: int = 35000,
) -> Ollama:
    """Returns Ollama instance for 'command-r' model."""
    return Ollama(
        name=name,
        endpoint=Endpoints.GENERATE.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Ollama.v1")
def ollama_command_r_plus(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "command-r-plus",
    strict: bool = Ollama.DEFAULT_STRICT,
    max_tries: int = Ollama.DEFAULT_MAX_TRIES,
    interval: float = Ollama.DEFAULT_INTERVAL,
    max_request_time: float = Ollama.DEFAULT_MAX_REQUEST_TIME,
    context_length: int = 35000,
) -> Ollama:
    """Returns Ollama instance for 'command-r-plus' model."""
    return Ollama(
        name=name,
        endpoint=Endpoints.GENERATE.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Ollama.v1")
def ollama_llava(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "llava",
    strict: bool = Ollama.DEFAULT_STRICT,
    max_tries: int = Ollama.DEFAULT_MAX_TRIES,
    interval: float = Ollama.DEFAULT_INTERVAL,
    max_request_time: float = Ollama.DEFAULT_MAX_REQUEST_TIME,
    context_length: int = 4096,
) -> Ollama:
    """Returns Ollama instance for 'llava' model."""
    return Ollama(
        name=name,
        endpoint=Endpoints.GENERATE.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Ollama.v1")
def ollama_dbrx(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "dbrx",
    strict: bool = Ollama.DEFAULT_STRICT,
    max_tries: int = Ollama.DEFAULT_MAX_TRIES,
    interval: float = Ollama.DEFAULT_INTERVAL,
    max_request_time: float = Ollama.DEFAULT_MAX_REQUEST_TIME,
    context_length: int = 4096,
) -> Ollama:
    """Returns Ollama instance for 'dbrx' model."""
    return Ollama(
        name=name,
        endpoint=Endpoints.GENERATE.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Ollama.v1")
def ollama_codellama(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "codellama",
    strict: bool = Ollama.DEFAULT_STRICT,
    max_tries: int = Ollama.DEFAULT_MAX_TRIES,
    interval: float = Ollama.DEFAULT_INTERVAL,
    max_request_time: float = Ollama.DEFAULT_MAX_REQUEST_TIME,
    context_length: int = 4096,
) -> Ollama:
    """Returns Ollama instance for 'codellama' model."""
    return Ollama(
        name=name,
        endpoint=Endpoints.GENERATE.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Ollama.v1")
def ollama_qwen(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "qwen",
    strict: bool = Ollama.DEFAULT_STRICT,
    max_tries: int = Ollama.DEFAULT_MAX_TRIES,
    interval: float = Ollama.DEFAULT_INTERVAL,
    max_request_time: float = Ollama.DEFAULT_MAX_REQUEST_TIME,
    context_length: int = 4096,
) -> Ollama:
    """Returns Ollama instance for 'qwen' model."""
    return Ollama(
        name=name,
        endpoint=Endpoints.GENERATE.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Ollama.v1")
def ollama_dolphin_mixtral(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "dolphin-mixtral",
    strict: bool = Ollama.DEFAULT_STRICT,
    max_tries: int = Ollama.DEFAULT_MAX_TRIES,
    interval: float = Ollama.DEFAULT_INTERVAL,
    max_request_time: float = Ollama.DEFAULT_MAX_REQUEST_TIME,
    context_length: int = 47000,
) -> Ollama:
    """Returns Ollama instance for 'dolphin-mixtral' model."""
    return Ollama(
        name=name,
        endpoint=Endpoints.GENERATE.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Ollama.v1")
def ollama_llama2_uncensored(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "llama2-uncensored",
    strict: bool = Ollama.DEFAULT_STRICT,
    max_tries: int = Ollama.DEFAULT_MAX_TRIES,
    interval: float = Ollama.DEFAULT_INTERVAL,
    max_request_time: float = Ollama.DEFAULT_MAX_REQUEST_TIME,
    context_length: int = 4096,
) -> Ollama:
    """Returns Ollama instance for 'llama2-uncensored' model."""
    return Ollama(
        name=name,
        endpoint=Endpoints.GENERATE.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Ollama.v1")
def ollama_mistral_openorca(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "mistral-openorca",
    strict: bool = Ollama.DEFAULT_STRICT,
    max_tries: int = Ollama.DEFAULT_MAX_TRIES,
    interval: float = Ollama.DEFAULT_INTERVAL,
    max_request_time: float = Ollama.DEFAULT_MAX_REQUEST_TIME,
    context_length: int = 4096,
) -> Ollama:
    """Returns Ollama instance for 'mistral-openorca' model."""
    return Ollama(
        name=name,
        endpoint=Endpoints.GENERATE.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Ollama.v1")
def ollama_deepseek_coder(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "deepseek-coder",
    strict: bool = Ollama.DEFAULT_STRICT,
    max_tries: int = Ollama.DEFAULT_MAX_TRIES,
    interval: float = Ollama.DEFAULT_INTERVAL,
    max_request_time: float = Ollama.DEFAULT_MAX_REQUEST_TIME,
    context_length: int = 4096,
) -> Ollama:
    """Returns Ollama instance for 'deepseek-coder' model."""
    return Ollama(
        name=name,
        endpoint=Endpoints.GENERATE.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Ollama.v1")
def ollama_phi(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "phi",
    strict: bool = Ollama.DEFAULT_STRICT,
    max_tries: int = Ollama.DEFAULT_MAX_TRIES,
    interval: float = Ollama.DEFAULT_INTERVAL,
    max_request_time: float = Ollama.DEFAULT_MAX_REQUEST_TIME,
    context_length: int = 4096,
) -> Ollama:
    """Returns Ollama instance for 'phi' model."""
    return Ollama(
        name=name,
        endpoint=Endpoints.GENERATE.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Ollama.v1")
def ollama_dolphin_mistral(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "dolphin-mistral",
    strict: bool = Ollama.DEFAULT_STRICT,
    max_tries: int = Ollama.DEFAULT_MAX_TRIES,
    interval: float = Ollama.DEFAULT_INTERVAL,
    max_request_time: float = Ollama.DEFAULT_MAX_REQUEST_TIME,
    context_length: int = 47000,
) -> Ollama:
    """Returns Ollama instance for 'dolphin-mistral' model."""
    return Ollama(
        name=name,
        endpoint=Endpoints.GENERATE.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Ollama.v1")
def ollama_nomic_embed_text(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "nomic-embed-text",
    strict: bool = Ollama.DEFAULT_STRICT,
    max_tries: int = Ollama.DEFAULT_MAX_TRIES,
    interval: float = Ollama.DEFAULT_INTERVAL,
    max_request_time: float = Ollama.DEFAULT_MAX_REQUEST_TIME,
    context_length: int = 4096,
) -> Ollama:
    """Returns Ollama instance for 'nomic-embed-text' model."""
    return Ollama(
        name=name,
        endpoint=Endpoints.GENERATE.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Ollama.v1")
def ollama_nous_hermes2(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "nous-hermes2",
    strict: bool = Ollama.DEFAULT_STRICT,
    max_tries: int = Ollama.DEFAULT_MAX_TRIES,
    interval: float = Ollama.DEFAULT_INTERVAL,
    max_request_time: float = Ollama.DEFAULT_MAX_REQUEST_TIME,
    context_length: int = 4096,
) -> Ollama:
    """Returns Ollama instance for 'nous-hermes2' model."""
    return Ollama(
        name=name,
        endpoint=Endpoints.GENERATE.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Ollama.v1")
def ollama_orca_mini(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "orca-mini",
    strict: bool = Ollama.DEFAULT_STRICT,
    max_tries: int = Ollama.DEFAULT_MAX_TRIES,
    interval: float = Ollama.DEFAULT_INTERVAL,
    max_request_time: float = Ollama.DEFAULT_MAX_REQUEST_TIME,
    context_length: int = 4096,
) -> Ollama:
    """Returns Ollama instance for 'orca-mini' model."""
    return Ollama(
        name=name,
        endpoint=Endpoints.GENERATE.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Ollama.v1")
def ollama_llama2_chinese(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "llama2-chinese",
    strict: bool = Ollama.DEFAULT_STRICT,
    max_tries: int = Ollama.DEFAULT_MAX_TRIES,
    interval: float = Ollama.DEFAULT_INTERVAL,
    max_request_time: float = Ollama.DEFAULT_MAX_REQUEST_TIME,
    context_length: int = 4096,
) -> Ollama:
    """Returns Ollama instance for 'llama2-chinese' model."""
    return Ollama(
        name=name,
        endpoint=Endpoints.GENERATE.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Ollama.v1")
def ollama_zephyr(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "zephyr",
    strict: bool = Ollama.DEFAULT_STRICT,
    max_tries: int = Ollama.DEFAULT_MAX_TRIES,
    interval: float = Ollama.DEFAULT_INTERVAL,
    max_request_time: float = Ollama.DEFAULT_MAX_REQUEST_TIME,
    context_length: int = 4096,
) -> Ollama:
    """Returns Ollama instance for 'zephyr' model."""
    return Ollama(
        name=name,
        endpoint=Endpoints.GENERATE.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Ollama.v1")
def ollama_wizard_vicuna_uncensored(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "wizard-vicuna-uncensored",
    strict: bool = Ollama.DEFAULT_STRICT,
    max_tries: int = Ollama.DEFAULT_MAX_TRIES,
    interval: float = Ollama.DEFAULT_INTERVAL,
    max_request_time: float = Ollama.DEFAULT_MAX_REQUEST_TIME,
    context_length: int = 4096,
) -> Ollama:
    """Returns Ollama instance for 'wizard-vicuna-uncensored' model."""
    return Ollama(
        name=name,
        endpoint=Endpoints.GENERATE.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Ollama.v1")
def ollama_openhermes(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "openhermes",
    strict: bool = Ollama.DEFAULT_STRICT,
    max_tries: int = Ollama.DEFAULT_MAX_TRIES,
    interval: float = Ollama.DEFAULT_INTERVAL,
    max_request_time: float = Ollama.DEFAULT_MAX_REQUEST_TIME,
    context_length: int = 4096,
) -> Ollama:
    """Returns Ollama instance for 'openhermes' model."""
    return Ollama(
        name=name,
        endpoint=Endpoints.GENERATE.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Ollama.v1")
def ollama_vicuna(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "vicuna",
    strict: bool = Ollama.DEFAULT_STRICT,
    max_tries: int = Ollama.DEFAULT_MAX_TRIES,
    interval: float = Ollama.DEFAULT_INTERVAL,
    max_request_time: float = Ollama.DEFAULT_MAX_REQUEST_TIME,
    context_length: int = 4096,
) -> Ollama:
    """Returns Ollama instance for 'vicuna' model."""
    return Ollama(
        name=name,
        endpoint=Endpoints.GENERATE.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Ollama.v1")
def ollama_tinyllama(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "tinyllama",
    strict: bool = Ollama.DEFAULT_STRICT,
    max_tries: int = Ollama.DEFAULT_MAX_TRIES,
    interval: float = Ollama.DEFAULT_INTERVAL,
    max_request_time: float = Ollama.DEFAULT_MAX_REQUEST_TIME,
    context_length: int = 4096,
) -> Ollama:
    """Returns Ollama instance for 'tinyllama' model."""
    return Ollama(
        name=name,
        endpoint=Endpoints.GENERATE.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Ollama.v1")
def ollama_tinydolphin(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "tinydolphin",
    strict: bool = Ollama.DEFAULT_STRICT,
    max_tries: int = Ollama.DEFAULT_MAX_TRIES,
    interval: float = Ollama.DEFAULT_INTERVAL,
    max_request_time: float = Ollama.DEFAULT_MAX_REQUEST_TIME,
    context_length: int = 4096,
) -> Ollama:
    """Returns Ollama instance for 'tinydolphin' model."""
    return Ollama(
        name=name,
        endpoint=Endpoints.GENERATE.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Ollama.v1")
def ollama_openchat(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "openchat",
    strict: bool = Ollama.DEFAULT_STRICT,
    max_tries: int = Ollama.DEFAULT_MAX_TRIES,
    interval: float = Ollama.DEFAULT_INTERVAL,
    max_request_time: float = Ollama.DEFAULT_MAX_REQUEST_TIME,
    context_length: int = 4096,
) -> Ollama:
    """Returns Ollama instance for 'openchat' model."""
    return Ollama(
        name=name,
        endpoint=Endpoints.GENERATE.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Ollama.v1")
def ollama_starcoder2(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "starcoder2",
    strict: bool = Ollama.DEFAULT_STRICT,
    max_tries: int = Ollama.DEFAULT_MAX_TRIES,
    interval: float = Ollama.DEFAULT_INTERVAL,
    max_request_time: float = Ollama.DEFAULT_MAX_REQUEST_TIME,
    context_length: int = 4096,
) -> Ollama:
    """Returns Ollama instance for 'starcoder2' model."""
    return Ollama(
        name=name,
        endpoint=Endpoints.GENERATE.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Ollama.v1")
def ollama_wizardcoder(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "wizardcoder",
    strict: bool = Ollama.DEFAULT_STRICT,
    max_tries: int = Ollama.DEFAULT_MAX_TRIES,
    interval: float = Ollama.DEFAULT_INTERVAL,
    max_request_time: float = Ollama.DEFAULT_MAX_REQUEST_TIME,
    context_length: int = 4096,
) -> Ollama:
    """Returns Ollama instance for 'wizardcoder' model."""
    return Ollama(
        name=name,
        endpoint=Endpoints.GENERATE.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Ollama.v1")
def ollama_stable_code(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "stable-code",
    strict: bool = Ollama.DEFAULT_STRICT,
    max_tries: int = Ollama.DEFAULT_MAX_TRIES,
    interval: float = Ollama.DEFAULT_INTERVAL,
    max_request_time: float = Ollama.DEFAULT_MAX_REQUEST_TIME,
    context_length: int = 4096,
) -> Ollama:
    """Returns Ollama instance for 'stable-code' model."""
    return Ollama(
        name=name,
        endpoint=Endpoints.GENERATE.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Ollama.v1")
def ollama_starcoder(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "starcoder",
    strict: bool = Ollama.DEFAULT_STRICT,
    max_tries: int = Ollama.DEFAULT_MAX_TRIES,
    interval: float = Ollama.DEFAULT_INTERVAL,
    max_request_time: float = Ollama.DEFAULT_MAX_REQUEST_TIME,
    context_length: int = 4096,
) -> Ollama:
    """Returns Ollama instance for 'starcoder' model."""
    return Ollama(
        name=name,
        endpoint=Endpoints.GENERATE.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Ollama.v1")
def ollama_neural_chat(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "neural-chat",
    strict: bool = Ollama.DEFAULT_STRICT,
    max_tries: int = Ollama.DEFAULT_MAX_TRIES,
    interval: float = Ollama.DEFAULT_INTERVAL,
    max_request_time: float = Ollama.DEFAULT_MAX_REQUEST_TIME,
    context_length: int = 4096,
) -> Ollama:
    """Returns Ollama instance for 'neural-chat' model."""
    return Ollama(
        name=name,
        endpoint=Endpoints.GENERATE.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Ollama.v1")
def ollama_yi(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "yi",
    strict: bool = Ollama.DEFAULT_STRICT,
    max_tries: int = Ollama.DEFAULT_MAX_TRIES,
    interval: float = Ollama.DEFAULT_INTERVAL,
    max_request_time: float = Ollama.DEFAULT_MAX_REQUEST_TIME,
    context_length: int = 4096,
) -> Ollama:
    """Returns Ollama instance for 'yi' model."""
    return Ollama(
        name=name,
        endpoint=Endpoints.GENERATE.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Ollama.v1")
def ollama_phind_codellama(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "phind-codellama",
    strict: bool = Ollama.DEFAULT_STRICT,
    max_tries: int = Ollama.DEFAULT_MAX_TRIES,
    interval: float = Ollama.DEFAULT_INTERVAL,
    max_request_time: float = Ollama.DEFAULT_MAX_REQUEST_TIME,
    context_length: int = 4096,
) -> Ollama:
    """Returns Ollama instance for 'phind-codellama' model."""
    return Ollama(
        name=name,
        endpoint=Endpoints.GENERATE.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Ollama.v1")
def ollama_starling_lm(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "starling-lm",
    strict: bool = Ollama.DEFAULT_STRICT,
    max_tries: int = Ollama.DEFAULT_MAX_TRIES,
    interval: float = Ollama.DEFAULT_INTERVAL,
    max_request_time: float = Ollama.DEFAULT_MAX_REQUEST_TIME,
    context_length: int = 4096,
) -> Ollama:
    """Returns Ollama instance for 'starling-lm' model."""
    return Ollama(
        name=name,
        endpoint=Endpoints.GENERATE.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Ollama.v1")
def ollama_wizard_math(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "wizard-math",
    strict: bool = Ollama.DEFAULT_STRICT,
    max_tries: int = Ollama.DEFAULT_MAX_TRIES,
    interval: float = Ollama.DEFAULT_INTERVAL,
    max_request_time: float = Ollama.DEFAULT_MAX_REQUEST_TIME,
    context_length: int = 4096,
) -> Ollama:
    """Returns Ollama instance for 'wizard-math' model."""
    return Ollama(
        name=name,
        endpoint=Endpoints.GENERATE.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Ollama.v1")
def ollama_falcon(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "falcon",
    strict: bool = Ollama.DEFAULT_STRICT,
    max_tries: int = Ollama.DEFAULT_MAX_TRIES,
    interval: float = Ollama.DEFAULT_INTERVAL,
    max_request_time: float = Ollama.DEFAULT_MAX_REQUEST_TIME,
    context_length: int = 4096,
) -> Ollama:
    """Returns Ollama instance for 'falcon' model."""
    return Ollama(
        name=name,
        endpoint=Endpoints.GENERATE.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Ollama.v1")
def ollama_dolphin_phi(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "dolphin-phi",
    strict: bool = Ollama.DEFAULT_STRICT,
    max_tries: int = Ollama.DEFAULT_MAX_TRIES,
    interval: float = Ollama.DEFAULT_INTERVAL,
    max_request_time: float = Ollama.DEFAULT_MAX_REQUEST_TIME,
    context_length: int = 4096,
) -> Ollama:
    """Returns Ollama instance for 'dolphin-phi' model."""
    return Ollama(
        name=name,
        endpoint=Endpoints.GENERATE.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Ollama.v1")
def ollama_orca2(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "orca2",
    strict: bool = Ollama.DEFAULT_STRICT,
    max_tries: int = Ollama.DEFAULT_MAX_TRIES,
    interval: float = Ollama.DEFAULT_INTERVAL,
    max_request_time: float = Ollama.DEFAULT_MAX_REQUEST_TIME,
    context_length: int = 4096,
) -> Ollama:
    """Returns Ollama instance for 'orca2' model."""
    return Ollama(
        name=name,
        endpoint=Endpoints.GENERATE.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Ollama.v1")
def ollama_dolphincoder(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "dolphincoder",
    strict: bool = Ollama.DEFAULT_STRICT,
    max_tries: int = Ollama.DEFAULT_MAX_TRIES,
    interval: float = Ollama.DEFAULT_INTERVAL,
    max_request_time: float = Ollama.DEFAULT_MAX_REQUEST_TIME,
    context_length: int = 4096,
) -> Ollama:
    """Returns Ollama instance for 'dolphincoder' model."""
    return Ollama(
        name=name,
        endpoint=Endpoints.GENERATE.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Ollama.v1")
def ollama_mxbai_embed_large(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "mxbai-embed-large",
    strict: bool = Ollama.DEFAULT_STRICT,
    max_tries: int = Ollama.DEFAULT_MAX_TRIES,
    interval: float = Ollama.DEFAULT_INTERVAL,
    max_request_time: float = Ollama.DEFAULT_MAX_REQUEST_TIME,
    context_length: int = 4096,
) -> Ollama:
    """Returns Ollama instance for 'mxbai-embed-large' model."""
    return Ollama(
        name=name,
        endpoint=Endpoints.GENERATE.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Ollama.v1")
def ollama_nous_hermes(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "nous-hermes",
    strict: bool = Ollama.DEFAULT_STRICT,
    max_tries: int = Ollama.DEFAULT_MAX_TRIES,
    interval: float = Ollama.DEFAULT_INTERVAL,
    max_request_time: float = Ollama.DEFAULT_MAX_REQUEST_TIME,
    context_length: int = 4096,
) -> Ollama:
    """Returns Ollama instance for 'nous-hermes' model."""
    return Ollama(
        name=name,
        endpoint=Endpoints.GENERATE.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Ollama.v1")
def ollama_solar(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "solar",
    strict: bool = Ollama.DEFAULT_STRICT,
    max_tries: int = Ollama.DEFAULT_MAX_TRIES,
    interval: float = Ollama.DEFAULT_INTERVAL,
    max_request_time: float = Ollama.DEFAULT_MAX_REQUEST_TIME,
    context_length: int = 4096,
) -> Ollama:
    """Returns Ollama instance for 'solar' model."""
    return Ollama(
        name=name,
        endpoint=Endpoints.GENERATE.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Ollama.v1")
def ollama_bakllava(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "bakllava",
    strict: bool = Ollama.DEFAULT_STRICT,
    max_tries: int = Ollama.DEFAULT_MAX_TRIES,
    interval: float = Ollama.DEFAULT_INTERVAL,
    max_request_time: float = Ollama.DEFAULT_MAX_REQUEST_TIME,
    context_length: int = 4096,
) -> Ollama:
    """Returns Ollama instance for 'bakllava' model."""
    return Ollama(
        name=name,
        endpoint=Endpoints.GENERATE.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Ollama.v1")
def ollama_sqlcoder(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "sqlcoder",
    strict: bool = Ollama.DEFAULT_STRICT,
    max_tries: int = Ollama.DEFAULT_MAX_TRIES,
    interval: float = Ollama.DEFAULT_INTERVAL,
    max_request_time: float = Ollama.DEFAULT_MAX_REQUEST_TIME,
    context_length: int = 4096,
) -> Ollama:
    """Returns Ollama instance for 'sqlcoder' model."""
    return Ollama(
        name=name,
        endpoint=Endpoints.GENERATE.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Ollama.v1")
def ollama_medllama2(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "medllama2",
    strict: bool = Ollama.DEFAULT_STRICT,
    max_tries: int = Ollama.DEFAULT_MAX_TRIES,
    interval: float = Ollama.DEFAULT_INTERVAL,
    max_request_time: float = Ollama.DEFAULT_MAX_REQUEST_TIME,
    context_length: int = 4096,
) -> Ollama:
    """Returns Ollama instance for 'medllama2' model."""
    return Ollama(
        name=name,
        endpoint=Endpoints.GENERATE.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Ollama.v1")
def ollama_nous_hermes2_mixtral(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "nous-hermes2-mixtral",
    strict: bool = Ollama.DEFAULT_STRICT,
    max_tries: int = Ollama.DEFAULT_MAX_TRIES,
    interval: float = Ollama.DEFAULT_INTERVAL,
    max_request_time: float = Ollama.DEFAULT_MAX_REQUEST_TIME,
    context_length: int = 47000,
) -> Ollama:
    """Returns Ollama instance for 'nous-hermes2-mixtral' model."""
    return Ollama(
        name=name,
        endpoint=Endpoints.GENERATE.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Ollama.v1")
def ollama_wizardlm_uncensored(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "wizardlm-uncensored",
    strict: bool = Ollama.DEFAULT_STRICT,
    max_tries: int = Ollama.DEFAULT_MAX_TRIES,
    interval: float = Ollama.DEFAULT_INTERVAL,
    max_request_time: float = Ollama.DEFAULT_MAX_REQUEST_TIME,
    context_length: int = 4096,
) -> Ollama:
    """Returns Ollama instance for 'wizardlm-uncensored' model."""
    return Ollama(
        name=name,
        endpoint=Endpoints.GENERATE.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Ollama.v1")
def ollama_dolphin_llama3(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "dolphin-llama3",
    strict: bool = Ollama.DEFAULT_STRICT,
    max_tries: int = Ollama.DEFAULT_MAX_TRIES,
    interval: float = Ollama.DEFAULT_INTERVAL,
    max_request_time: float = Ollama.DEFAULT_MAX_REQUEST_TIME,
    context_length: int = 4096,
) -> Ollama:
    """Returns Ollama instance for 'dolphin-llama3' model."""
    return Ollama(
        name=name,
        endpoint=Endpoints.GENERATE.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Ollama.v1")
def ollama_codeup(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "codeup",
    strict: bool = Ollama.DEFAULT_STRICT,
    max_tries: int = Ollama.DEFAULT_MAX_TRIES,
    interval: float = Ollama.DEFAULT_INTERVAL,
    max_request_time: float = Ollama.DEFAULT_MAX_REQUEST_TIME,
    context_length: int = 4096,
) -> Ollama:
    """Returns Ollama instance for 'codeup' model."""
    return Ollama(
        name=name,
        endpoint=Endpoints.GENERATE.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Ollama.v1")
def ollama_stablelm2(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "stablelm2",
    strict: bool = Ollama.DEFAULT_STRICT,
    max_tries: int = Ollama.DEFAULT_MAX_TRIES,
    interval: float = Ollama.DEFAULT_INTERVAL,
    max_request_time: float = Ollama.DEFAULT_MAX_REQUEST_TIME,
    context_length: int = 4096,
) -> Ollama:
    """Returns Ollama instance for 'stablelm2' model."""
    return Ollama(
        name=name,
        endpoint=Endpoints.GENERATE.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Ollama.v1")
def ollama_everythinglm(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "everythinglm",
    strict: bool = Ollama.DEFAULT_STRICT,
    max_tries: int = Ollama.DEFAULT_MAX_TRIES,
    interval: float = Ollama.DEFAULT_INTERVAL,
    max_request_time: float = Ollama.DEFAULT_MAX_REQUEST_TIME,
    context_length: int = 16384,
) -> Ollama:
    """Returns Ollama instance for 'everythinglm' model."""
    return Ollama(
        name=name,
        endpoint=Endpoints.GENERATE.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Ollama.v1")
def ollama_all_minilm(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "all-minilm",
    strict: bool = Ollama.DEFAULT_STRICT,
    max_tries: int = Ollama.DEFAULT_MAX_TRIES,
    interval: float = Ollama.DEFAULT_INTERVAL,
    max_request_time: float = Ollama.DEFAULT_MAX_REQUEST_TIME,
    context_length: int = 4096,
) -> Ollama:
    """Returns Ollama instance for 'all-minilm' model."""
    return Ollama(
        name=name,
        endpoint=Endpoints.GENERATE.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Ollama.v1")
def ollama_samantha_mistral(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "samantha-mistral",
    strict: bool = Ollama.DEFAULT_STRICT,
    max_tries: int = Ollama.DEFAULT_MAX_TRIES,
    interval: float = Ollama.DEFAULT_INTERVAL,
    max_request_time: float = Ollama.DEFAULT_MAX_REQUEST_TIME,
    context_length: int = 4096,
) -> Ollama:
    """Returns Ollama instance for 'samantha-mistral' model."""
    return Ollama(
        name=name,
        endpoint=Endpoints.GENERATE.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Ollama.v1")
def ollama_yarn_mistral(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "yarn-mistral",
    strict: bool = Ollama.DEFAULT_STRICT,
    max_tries: int = Ollama.DEFAULT_MAX_TRIES,
    interval: float = Ollama.DEFAULT_INTERVAL,
    max_request_time: float = Ollama.DEFAULT_MAX_REQUEST_TIME,
    context_length: int = 128000,
) -> Ollama:
    """Returns Ollama instance for 'yarn-mistral' model."""
    return Ollama(
        name=name,
        endpoint=Endpoints.GENERATE.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Ollama.v1")
def ollama_stable_beluga(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "stable-beluga",
    strict: bool = Ollama.DEFAULT_STRICT,
    max_tries: int = Ollama.DEFAULT_MAX_TRIES,
    interval: float = Ollama.DEFAULT_INTERVAL,
    max_request_time: float = Ollama.DEFAULT_MAX_REQUEST_TIME,
    context_length: int = 4096,
) -> Ollama:
    """Returns Ollama instance for 'stable-beluga' model."""
    return Ollama(
        name=name,
        endpoint=Endpoints.GENERATE.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Ollama.v1")
def ollama_meditron(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "meditron",
    strict: bool = Ollama.DEFAULT_STRICT,
    max_tries: int = Ollama.DEFAULT_MAX_TRIES,
    interval: float = Ollama.DEFAULT_INTERVAL,
    max_request_time: float = Ollama.DEFAULT_MAX_REQUEST_TIME,
    context_length: int = 4096,
) -> Ollama:
    """Returns Ollama instance for 'meditron' model."""
    return Ollama(
        name=name,
        endpoint=Endpoints.GENERATE.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Ollama.v1")
def ollama_yarn_llama2(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "yarn-llama2",
    strict: bool = Ollama.DEFAULT_STRICT,
    max_tries: int = Ollama.DEFAULT_MAX_TRIES,
    interval: float = Ollama.DEFAULT_INTERVAL,
    max_request_time: float = Ollama.DEFAULT_MAX_REQUEST_TIME,
    context_length: int = 128000,
) -> Ollama:
    """Returns Ollama instance for 'yarn-llama2' model."""
    return Ollama(
        name=name,
        endpoint=Endpoints.GENERATE.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Ollama.v1")
def ollama_deepseek_llm(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "deepseek-llm",
    strict: bool = Ollama.DEFAULT_STRICT,
    max_tries: int = Ollama.DEFAULT_MAX_TRIES,
    interval: float = Ollama.DEFAULT_INTERVAL,
    max_request_time: float = Ollama.DEFAULT_MAX_REQUEST_TIME,
    context_length: int = 4096,
) -> Ollama:
    """Returns Ollama instance for 'deepseek-llm' model."""
    return Ollama(
        name=name,
        endpoint=Endpoints.GENERATE.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Ollama.v1")
def ollama_llama_pro(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "llama-pro",
    strict: bool = Ollama.DEFAULT_STRICT,
    max_tries: int = Ollama.DEFAULT_MAX_TRIES,
    interval: float = Ollama.DEFAULT_INTERVAL,
    max_request_time: float = Ollama.DEFAULT_MAX_REQUEST_TIME,
    context_length: int = 4096,
) -> Ollama:
    """Returns Ollama instance for 'llama-pro' model."""
    return Ollama(
        name=name,
        endpoint=Endpoints.GENERATE.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Ollama.v1")
def ollama_magicoder(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "magicoder",
    strict: bool = Ollama.DEFAULT_STRICT,
    max_tries: int = Ollama.DEFAULT_MAX_TRIES,
    interval: float = Ollama.DEFAULT_INTERVAL,
    max_request_time: float = Ollama.DEFAULT_MAX_REQUEST_TIME,
    context_length: int = 4096,
) -> Ollama:
    """Returns Ollama instance for 'magicoder' model."""
    return Ollama(
        name=name,
        endpoint=Endpoints.GENERATE.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Ollama.v1")
def ollama_stablelm_zephyr(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "stablelm-zephyr",
    strict: bool = Ollama.DEFAULT_STRICT,
    max_tries: int = Ollama.DEFAULT_MAX_TRIES,
    interval: float = Ollama.DEFAULT_INTERVAL,
    max_request_time: float = Ollama.DEFAULT_MAX_REQUEST_TIME,
    context_length: int = 4096,
) -> Ollama:
    """Returns Ollama instance for 'stablelm-zephyr' model."""
    return Ollama(
        name=name,
        endpoint=Endpoints.GENERATE.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Ollama.v1")
def ollama_codebooga(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "codebooga",
    strict: bool = Ollama.DEFAULT_STRICT,
    max_tries: int = Ollama.DEFAULT_MAX_TRIES,
    interval: float = Ollama.DEFAULT_INTERVAL,
    max_request_time: float = Ollama.DEFAULT_MAX_REQUEST_TIME,
    context_length: int = 4096,
) -> Ollama:
    """Returns Ollama instance for 'codebooga' model."""
    return Ollama(
        name=name,
        endpoint=Endpoints.GENERATE.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Ollama.v1")
def ollama_codeqwen(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "codeqwen",
    strict: bool = Ollama.DEFAULT_STRICT,
    max_tries: int = Ollama.DEFAULT_MAX_TRIES,
    interval: float = Ollama.DEFAULT_INTERVAL,
    max_request_time: float = Ollama.DEFAULT_MAX_REQUEST_TIME,
    context_length: int = 4096,
) -> Ollama:
    """Returns Ollama instance for 'codeqwen' model."""
    return Ollama(
        name=name,
        endpoint=Endpoints.GENERATE.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Ollama.v1")
def ollama_mistrallite(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "mistrallite",
    strict: bool = Ollama.DEFAULT_STRICT,
    max_tries: int = Ollama.DEFAULT_MAX_TRIES,
    interval: float = Ollama.DEFAULT_INTERVAL,
    max_request_time: float = Ollama.DEFAULT_MAX_REQUEST_TIME,
    context_length: int = 8192,
) -> Ollama:
    """Returns Ollama instance for 'mistrallite' model."""
    return Ollama(
        name=name,
        endpoint=Endpoints.GENERATE.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Ollama.v1")
def ollama_wizard_vicuna(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "wizard-vicuna",
    strict: bool = Ollama.DEFAULT_STRICT,
    max_tries: int = Ollama.DEFAULT_MAX_TRIES,
    interval: float = Ollama.DEFAULT_INTERVAL,
    max_request_time: float = Ollama.DEFAULT_MAX_REQUEST_TIME,
    context_length: int = 4096,
) -> Ollama:
    """Returns Ollama instance for 'wizard-vicuna' model."""
    return Ollama(
        name=name,
        endpoint=Endpoints.GENERATE.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Ollama.v1")
def ollama_nexusraven(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "nexusraven",
    strict: bool = Ollama.DEFAULT_STRICT,
    max_tries: int = Ollama.DEFAULT_MAX_TRIES,
    interval: float = Ollama.DEFAULT_INTERVAL,
    max_request_time: float = Ollama.DEFAULT_MAX_REQUEST_TIME,
    context_length: int = 4096,
) -> Ollama:
    """Returns Ollama instance for 'nexusraven' model."""
    return Ollama(
        name=name,
        endpoint=Endpoints.GENERATE.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Ollama.v1")
def ollama_xwinlm(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "xwinlm",
    strict: bool = Ollama.DEFAULT_STRICT,
    max_tries: int = Ollama.DEFAULT_MAX_TRIES,
    interval: float = Ollama.DEFAULT_INTERVAL,
    max_request_time: float = Ollama.DEFAULT_MAX_REQUEST_TIME,
    context_length: int = 4096,
) -> Ollama:
    """Returns Ollama instance for 'xwinlm' model."""
    return Ollama(
        name=name,
        endpoint=Endpoints.GENERATE.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Ollama.v1")
def ollama_goliath(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "goliath",
    strict: bool = Ollama.DEFAULT_STRICT,
    max_tries: int = Ollama.DEFAULT_MAX_TRIES,
    interval: float = Ollama.DEFAULT_INTERVAL,
    max_request_time: float = Ollama.DEFAULT_MAX_REQUEST_TIME,
    context_length: int = 4096,
) -> Ollama:
    """Returns Ollama instance for 'goliath' model."""
    return Ollama(
        name=name,
        endpoint=Endpoints.GENERATE.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Ollama.v1")
def ollama_open_orca_platypus2(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "open-orca-platypus2",
    strict: bool = Ollama.DEFAULT_STRICT,
    max_tries: int = Ollama.DEFAULT_MAX_TRIES,
    interval: float = Ollama.DEFAULT_INTERVAL,
    max_request_time: float = Ollama.DEFAULT_MAX_REQUEST_TIME,
    context_length: int = 4096,
) -> Ollama:
    """Returns Ollama instance for 'open-orca-platypus2' model."""
    return Ollama(
        name=name,
        endpoint=Endpoints.GENERATE.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Ollama.v1")
def ollama_wizardlm(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "wizardlm",
    strict: bool = Ollama.DEFAULT_STRICT,
    max_tries: int = Ollama.DEFAULT_MAX_TRIES,
    interval: float = Ollama.DEFAULT_INTERVAL,
    max_request_time: float = Ollama.DEFAULT_MAX_REQUEST_TIME,
    context_length: int = 4096,
) -> Ollama:
    """Returns Ollama instance for 'wizardlm' model."""
    return Ollama(
        name=name,
        endpoint=Endpoints.GENERATE.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Ollama.v1")
def ollama_notux(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "notux",
    strict: bool = Ollama.DEFAULT_STRICT,
    max_tries: int = Ollama.DEFAULT_MAX_TRIES,
    interval: float = Ollama.DEFAULT_INTERVAL,
    max_request_time: float = Ollama.DEFAULT_MAX_REQUEST_TIME,
    context_length: int = 4096,
) -> Ollama:
    """Returns Ollama instance for 'notux' model."""
    return Ollama(
        name=name,
        endpoint=Endpoints.GENERATE.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Ollama.v1")
def ollama_megadolphin(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "megadolphin",
    strict: bool = Ollama.DEFAULT_STRICT,
    max_tries: int = Ollama.DEFAULT_MAX_TRIES,
    interval: float = Ollama.DEFAULT_INTERVAL,
    max_request_time: float = Ollama.DEFAULT_MAX_REQUEST_TIME,
    context_length: int = 4096,
) -> Ollama:
    """Returns Ollama instance for 'megadolphin' model."""
    return Ollama(
        name=name,
        endpoint=Endpoints.GENERATE.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Ollama.v1")
def ollama_duckdb_nsql(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "duckdb-nsql",
    strict: bool = Ollama.DEFAULT_STRICT,
    max_tries: int = Ollama.DEFAULT_MAX_TRIES,
    interval: float = Ollama.DEFAULT_INTERVAL,
    max_request_time: float = Ollama.DEFAULT_MAX_REQUEST_TIME,
    context_length: int = 4096,
) -> Ollama:
    """Returns Ollama instance for 'duckdb-nsql' model."""
    return Ollama(
        name=name,
        endpoint=Endpoints.GENERATE.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Ollama.v1")
def ollama_alfred(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "alfred",
    strict: bool = Ollama.DEFAULT_STRICT,
    max_tries: int = Ollama.DEFAULT_MAX_TRIES,
    interval: float = Ollama.DEFAULT_INTERVAL,
    max_request_time: float = Ollama.DEFAULT_MAX_REQUEST_TIME,
    context_length: int = 4096,
) -> Ollama:
    """Returns Ollama instance for 'alfred' model."""
    return Ollama(
        name=name,
        endpoint=Endpoints.GENERATE.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Ollama.v1")
def ollama_notus(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "notus",
    strict: bool = Ollama.DEFAULT_STRICT,
    max_tries: int = Ollama.DEFAULT_MAX_TRIES,
    interval: float = Ollama.DEFAULT_INTERVAL,
    max_request_time: float = Ollama.DEFAULT_MAX_REQUEST_TIME,
    context_length: int = 4096,
) -> Ollama:
    """Returns Ollama instance for 'notus' model."""
    return Ollama(
        name=name,
        endpoint=Endpoints.GENERATE.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )


@registry.llm_models("spacy.Ollama.v1")
def ollama_snowflake_arctic_embed(
    config: Dict[Any, Any] = SimpleFrozenDict(),
    name: str = "snowflake-arctic-embed",
    strict: bool = Ollama.DEFAULT_STRICT,
    max_tries: int = Ollama.DEFAULT_MAX_TRIES,
    interval: float = Ollama.DEFAULT_INTERVAL,
    max_request_time: float = Ollama.DEFAULT_MAX_REQUEST_TIME,
    context_length: int = 4096,
) -> Ollama:
    """Returns Ollama instance for 'snowflake-arctic-embed' model."""
    return Ollama(
        name=name,
        endpoint=Endpoints.GENERATE.value,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        context_length=context_length,
    )
