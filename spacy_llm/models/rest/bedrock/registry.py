from typing import Any, Callable, Dict, Iterable, List

from confection import SimpleFrozenDict

from ....registry import registry
from .model import Bedrock, Models

_DEFAULT_RETRIES: int = 5
_DEFAULT_TEMPERATURE: float = 0.0
_DEFAULT_MAX_TOKEN_COUNT: int = 512
_DEFAULT_TOP_P: int = 1
_DEFAULT_STOP_SEQUENCES: List[str] = []
_DEFAULT_COUNT_PENALTY: Dict[str, Any] = {"scale": 0}
_DEFAULT_PRESENCE_PENALTY: Dict[str, Any] = {"scale": 0}
_DEFAULT_FREQUENCY_PENALTY: Dict[str, Any] = {"scale": 0}


@registry.llm_models("spacy.Bedrock.v1")
def bedrock(
    region: str,
    model_id: Models = Models.TITAN_EXPRESS,
    config: Dict[Any, Any] = SimpleFrozenDict(
        # Params for Titan models
        temperature=_DEFAULT_TEMPERATURE,
        maxTokenCount=_DEFAULT_MAX_TOKEN_COUNT,
        stopSequences=_DEFAULT_STOP_SEQUENCES,
        topP=_DEFAULT_TOP_P,
        # Params for Jurassic models
        maxTokens=_DEFAULT_MAX_TOKEN_COUNT,
        countPenalty=_DEFAULT_COUNT_PENALTY,
        presencePenalty=_DEFAULT_PRESENCE_PENALTY,
        frequencyPenalty=_DEFAULT_FREQUENCY_PENALTY,
        stop_sequences=_DEFAULT_STOP_SEQUENCES,
    ),
    max_tries: int = _DEFAULT_RETRIES,
) -> Callable[[Iterable[str]], Iterable[str]]:
    """Returns Bedrock instance for 'amazon-titan-express' model using boto3 to prompt API.
    model_id (ModelId): ID of the deployed model (titan-express)
    region (str): Specify the AWS region for the service
    config (Dict[Any, Any]): LLM config passed on to the model's initialization.
    """
    return Bedrock(model_id=model_id, region=region, config=config, max_tries=max_tries)
