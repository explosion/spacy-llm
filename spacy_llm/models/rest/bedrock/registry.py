from typing import Any, Callable, Dict, Iterable, List

from confection import SimpleFrozenDict

from ...registry import registry
from .model import Bedrock, Models

_DEFAULT_RETRIES: int = 5
_DEFAULT_TEMPERATURE: float = 0.0
_DEFAULT_MAX_TOKEN_COUNT: int = 512
_DEFAULT_TOP_P: int = 1
_DEFAULT_STOP_SEQUENCES: List[str] = []


@registry.llm_models("spacy.Bedrock.Titan.Express.v1")
def titan_express(
    region: str,
    model_id: Models = Models.TITAN_EXPRESS,
    config: Dict[Any, Any] = SimpleFrozenDict(
        temperature=_DEFAULT_TEMPERATURE,
        maxTokenCount=_DEFAULT_MAX_TOKEN_COUNT,
        stopSequences=_DEFAULT_STOP_SEQUENCES,
        topP=_DEFAULT_TOP_P,
    ),
    max_retries: int = _DEFAULT_RETRIES,
) -> Callable[[Iterable[str]], Iterable[str]]:
    """Returns Bedrock instance for 'amazon-titan-express' model using boto3 to prompt API.
    model_id (ModelId): ID of the deployed model (titan-express)
    region (str): Specify the AWS region for the service
    config (Dict[Any, Any]): LLM config passed on to the model's initialization.
    """
    return Bedrock(
        model_id=model_id, region=region, config=config, max_retries=max_retries
    )


@registry.llm_models("spacy.Bedrock.Titan.Lite.v1")
def titan_lite(
    region: str,
    model_id: Models = Models.TITAN_LITE,
    config: Dict[Any, Any] = SimpleFrozenDict(
        temperature=_DEFAULT_TEMPERATURE,
        maxTokenCount=_DEFAULT_MAX_TOKEN_COUNT,
        stopSequences=_DEFAULT_STOP_SEQUENCES,
        topP=_DEFAULT_TOP_P,
    ),
    max_retries: int = _DEFAULT_RETRIES,
) -> Callable[[Iterable[str]], Iterable[str]]:
    """Returns Bedrock instance for 'amazon-titan-lite' model using boto3 to prompt API.
    region (str): Specify the AWS region for the service
    model_id (ModelId): ID of the deployed model (titan-lite)
    region (str): Specify the AWS region for the service
    config (Dict[Any, Any]): LLM config passed on to the model's initialization.
    """
    return Bedrock(
        model_id=model_id,
        region=region,
        config=config,
        max_retries=max_retries,
    )
