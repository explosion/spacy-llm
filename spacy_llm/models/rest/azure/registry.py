from typing import Any, Callable, Dict, Iterable, Optional

from confection import SimpleFrozenDict

from ....registry import registry
from .model import AzureOpenAI, ModelType

_DEFAULT_TEMPERATURE = 0.0


@registry.llm_models("spacy.Azure.v2")
def azure_openai_v2(
    deployment_name: str,
    name: str,
    base_url: str,
    model_type: ModelType,
    config: Dict[Any, Any] = SimpleFrozenDict(temperature=_DEFAULT_TEMPERATURE),
    strict: bool = AzureOpenAI.DEFAULT_STRICT,
    max_tries: int = AzureOpenAI.DEFAULT_MAX_TRIES,
    interval: float = AzureOpenAI.DEFAULT_INTERVAL,
    max_request_time: float = AzureOpenAI.DEFAULT_MAX_REQUEST_TIME,
    api_version: str = "2023-05-15",
    context_length: Optional[int] = None,
) -> Callable[[Iterable[Iterable[str]]], Iterable[Iterable[str]]]:
    """Returns Azure OpenAI instance for models deployed on Azure's OpenAI service using REST to prompt API.

    Docs on OpenAI models supported by Azure:
    https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/models#model-summary-table-and-region-availability.

    config (Dict[Any, Any]): LLM config passed on to the model's initialization.
    deployment_name (str): Name of the deployment to use. Note that this does not necessarily equal the name of the
        model used by that deployment, as deployment names in Azure OpenAI can be arbitrary.
    name (str): Name of the model used by this deployment. This is required to infer the context length that can be
        assumed for prompting.
    endpoint (str): The URL for your Azure OpenAI endpoint. This is usually something like
        "https://{prefix}.openai.azure.com/".
    model_type (ModelType): Whether the deployed model is a text completetion model (e. g.
        text-davinci-003) or a chat model (e. g. gpt-4).
    strict (bool): If True, ValueError is raised if the LLM API returns a malformed response (i. e. any kind of JSON
        or other response object that does not conform to the expectation of how a well-formed response object from
        this API should look like). If False, the API error responses are returned by __call__(), but no error will
        be raised.
    max_tries (int): Max. number of tries for API request.
    interval (float): Time interval (in seconds) for API retries in seconds. We implement a base 2 exponential backoff
        at each retry.
    max_request_time (float): Max. time (in seconds) to wait for request to terminate before raising an exception.
    api_version (str): API version to use.
    context_length (Optional[int]): Context length for this model. Only necessary for sharding and if no context length
        natively provided by spacy-llm.
    RETURNS (AzureOpenAI): AzureOpenAI instance for deployed model.

    DOCS: https://spacy.io/api/large-language-models#models
    """
    return AzureOpenAI(
        deployment_name=deployment_name,
        name=name,
        endpoint=base_url,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        api_version=api_version,
        model_type=model_type,
        context_length=context_length,
    )


@registry.llm_models("spacy.Azure.v1")
def azure_openai(
    deployment_name: str,
    name: str,
    base_url: str,
    model_type: ModelType,
    config: Dict[Any, Any] = SimpleFrozenDict(temperature=_DEFAULT_TEMPERATURE),
    strict: bool = AzureOpenAI.DEFAULT_STRICT,
    max_tries: int = AzureOpenAI.DEFAULT_MAX_TRIES,
    interval: float = AzureOpenAI.DEFAULT_INTERVAL,
    max_request_time: float = AzureOpenAI.DEFAULT_MAX_REQUEST_TIME,
    api_version: str = "2023-05-15",
) -> Callable[[Iterable[Iterable[str]]], Iterable[Iterable[str]]]:
    """Returns Azure OpenAI instance for models deployed on Azure's OpenAI service using REST to prompt API.

    Docs on OpenAI models supported by Azure:
    https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/models#model-summary-table-and-region-availability.

    config (Dict[Any, Any]): LLM config passed on to the model's initialization.
    deployment_name (str): Name of the deployment to use. Note that this does not necessarily equal the name of the
        model used by that deployment, as deployment names in Azure OpenAI can be arbitrary.
    name (str): Name of the model used by this deployment. This is required to infer the context length that can be
        assumed for prompting.
    endpoint (str): The URL for your Azure OpenAI endpoint. This is usually something like
        "https://{prefix}.openai.azure.com/".
    model_type (ModelType): Whether the deployed model is a text completetion model (e. g.
        text-davinci-003) or a chat model (e. g. gpt-4).
    strict (bool): If True, ValueError is raised if the LLM API returns a malformed response (i. e. any kind of JSON
        or other response object that does not conform to the expectation of how a well-formed response object from
        this API should look like). If False, the API error responses are returned by __call__(), but no error will
        be raised.
    max_tries (int): Max. number of tries for API request.
    interval (float): Time interval (in seconds) for API retries in seconds. We implement a base 2 exponential backoff
        at each retry.
    max_request_time (float): Max. time (in seconds) to wait for request to terminate before raising an exception.
    api_version (str): API version to use.
    RETURNS (AzureOpenAI): AzureOpenAI instance for deployed model.

    DOCS: https://spacy.io/api/large-language-models#models
    """
    return AzureOpenAI(
        deployment_name=deployment_name,
        name=name,
        endpoint=base_url,
        config=config,
        strict=strict,
        max_tries=max_tries,
        interval=interval,
        max_request_time=max_request_time,
        api_version=api_version,
        model_type=model_type,
        context_length=None,
    )
