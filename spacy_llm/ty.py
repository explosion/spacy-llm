import abc
import inspect
import typing
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Generic, Iterable, List, Optional, Tuple, Type
from typing import TypeVar, Union, cast

from spacy.tokens import Doc
from spacy.training.example import Example
from spacy.vocab import Vocab

from .compat import GenericModel, Protocol, Self, runtime_checkable
from .models import langchain

_PromptType = Any
_ResponseType = Any
_ParsedResponseType = Any

PromptExecutorType = Callable[
    [Iterable[Iterable[_PromptType]]], Iterable[_ResponseType]
]
ExamplesConfigType = Union[
    Iterable[Dict[str, Any]], Callable[[], Iterable[Dict[str, Any]]], None
]
NTokenEstimator = Callable[[str], int]
ShardMapper = Callable[
    # Requires doc, doc index, context length and callable for rendering template from doc shard text.
    [Doc, int, int, Callable[[Doc, int, int, int], str]],
    # Returns each shard as a doc.
    Iterable[Doc],
]


@runtime_checkable
class Serializable(Protocol):
    def to_bytes(
        self,
        *,
        exclude: Tuple[str] = cast(Tuple[str], tuple()),
    ) -> bytes:
        ...

    def from_bytes(
        self,
        bytes_data: bytes,
        *,
        exclude: Tuple[str] = cast(Tuple[str], tuple()),
    ) -> bytes:
        ...

    def to_disk(
        self,
        path: Path,
        *,
        exclude: Tuple[str] = cast(Tuple[str], tuple()),
    ) -> bytes:
        ...

    def from_disk(
        self,
        path: Path,
        *,
        exclude: Tuple[str] = cast(Tuple[str], tuple()),
    ):
        ...


@runtime_checkable
class ScorableTask(Protocol):
    """Differs from Scorable in that it describes an object with a scorer() function, i. e. a scorable
    as checked for by the LLMWrapper component.
    """

    def scorer(
        self,
        examples: Iterable[Example],
    ) -> Dict[str, Any]:
        """Scores performance on examples."""


@runtime_checkable
class Scorer(Protocol):
    """Differs from ScorableTask in that it describes a Callable with a call signature matching the scorer()
    function + kwargs, i. e. a scorable as passed via the configuration.
    """

    def __call__(self, examples: Iterable[Example], **kwargs) -> Dict[str, Any]:
        """Score performance on examples.
        examples (Iterable[Example]): Examples to score.
        RETURNS (Dict[str, Any]): Dict with metric name -> score.
        """


@runtime_checkable
class ShardingLLMTask(Protocol):
    def generate_prompts(
        self, docs: Iterable[Doc], context_length: Optional[int] = None
    ) -> Iterable[Tuple[Iterable[_PromptType], Iterable[Doc]]]:
        """Generate prompts from docs.
        docs (Iterable[Doc]): Docs to generate prompts from.
        context_length (int): Context length for model this task is executed with. Needed for sharding and fusing docs,
            if the corresponding prompts exceed the context length. If None, context length is assumed to be infinite.
        RETURNS (Iterable[Tuple[Iterable[_PromptType], Iterable[Doc]]]): Iterable with one to n prompts per doc
            (multiple prompts in case of multiple shards) and the corresponding shards. The relationship between shard
            and prompt is 1:1.
        """

    def parse_responses(
        self,
        shards: Iterable[Iterable[Doc]],
        responses: Iterable[Iterable[_ResponseType]],
    ) -> Iterable[Doc]:
        """
        Parses LLM responses.
        docs (Iterable[Iterable[Doc]]): Doc shards to map responses into.
        responses ([Iterable[Iterable[_ResponseType]]]): LLM responses.
        RETURNS (Iterable[Doc]]): Updated (and fused) docs.
        """


@runtime_checkable
class NonshardingLLMTask(Protocol):
    def generate_prompts(self, docs: Iterable[Doc]) -> Iterable[_PromptType]:
        """Generate prompts from docs.
        docs (Iterable[Doc]): Docs to generate prompts from.
        RETURNS (Iterable[_PromptType]): Iterable with one prompt per doc.
        """

    def parse_responses(
        self, docs: Iterable[Doc], responses: Iterable[_ResponseType]
    ) -> Iterable[Doc]:
        """
        Parses LLM responses.
        docs (Iterable[Doc]): Docs to map responses into.
        responses ([Iterable[_ResponseType]]): LLM responses.
        RETURNS (Iterable[Doc]]): Updated docs.
        """


LLMTask = Union[NonshardingLLMTask, ShardingLLMTask]

TaskContraT = TypeVar(
    "TaskContraT", bound=Union[ShardingLLMTask, NonshardingLLMTask], contravariant=True
)
ShardingTaskContraT = TypeVar(
    "ShardingTaskContraT", bound=ShardingLLMTask, contravariant=True
)


@runtime_checkable
class ShardReducer(Protocol[ShardingTaskContraT]):
    """Generic protocol for tasks' shard reducer."""

    def __call__(self, task: ShardingTaskContraT, shards: Iterable[Doc]) -> Doc:
        """Merges shard to single Doc."""
        ...


class FewshotExample(GenericModel, abc.ABC, Generic[TaskContraT]):
    @classmethod
    @abc.abstractmethod
    def generate(cls, example: Example, task: TaskContraT) -> Optional[Self]:
        """Create a fewshot example from a spaCy example.
        example (Example): spaCy example.
        task (TaskContraT): Task for which to generate examples.
        RETURNS (Optional[Self]): Generated example. None, if example couldn't be generated.
        """


class TaskResponseParser(Protocol[TaskContraT]):
    """Generic protocol for parsing functions with specific tasks."""

    def __call__(
        self,
        task: TaskContraT,
        shards: Iterable[Iterable[Doc]],
        responses: Iterable[Iterable[Any]],
    ) -> Iterable[Iterable[Any]]:
        ...


@runtime_checkable
class PromptTemplateProvider(Protocol):
    @property
    def prompt_template(self) -> str:
        ...


@runtime_checkable
class LabeledTask(Protocol):
    @property
    def labels(self) -> Tuple[str, ...]:
        ...

    def add_label(self, label: str, label_definition: Optional[str] = None) -> int:
        ...

    def clear(self) -> None:
        ...


@runtime_checkable
class Cache(Protocol):
    """Defines minimal set of operations a cache implementiation needs to support."""

    def initialize(
        self, vocab: Vocab, task: Union[NonshardingLLMTask, ShardingLLMTask]
    ) -> None:
        """
        Initialize cache with data not available at construction time.
        vocab (Vocab): Vocab object.
        task (Union[LLMTask, ShardingLLMTask]): Task.
        """

    def add(self, doc: Doc) -> None:
        """Adds processed doc to cache (or to a queue that is added to the cache at a later point)
        doc (Doc): Doc to add to persistence queue.
        """

    @property
    def prompt_template(self) -> Optional[str]:
        """Get prompt template.
        RETURNS (Optional[str]): Prompt template string used for docs to cache/cached docs.
        """

    @prompt_template.setter
    def prompt_template(self, prompt_template: str) -> None:
        """Set prompt template.
        prompt_template (str): Prompt template string used for docs to cache/cached docs.
        """

    def __contains__(self, doc: Doc) -> bool:
        """Checks whether doc has been processed and cached.
        doc (Doc): Doc to check for.
        RETURNS (bool): Whether doc has been processed and cached.
        """

    def __getitem__(self, doc: Doc) -> Optional[Doc]:
        """Loads doc from cache. If doc is not in cache, None is returned.
        doc (Doc): Unprocessed doc whose processed equivalent should be returned.
        RETURNS (Optional[Doc]): Cached and processed version of doc, if available. Otherwise None.
        """


@runtime_checkable
class ModelWithContextLength(Protocol):
    @property
    def context_length(self) -> int:
        """Provides context length for the corresponding model.
        RETURNS (int): Context length for the corresponding model.
        """


def _do_args_match(out_arg: Iterable, in_arg: Iterable, nesting_level: int) -> bool:
    """Compares argument type of Iterables for compatibility.
    in_arg (Iterable): Input argument.
    out_arg (Iterable): Output argument.
    nesting_level (int): Expected level of nesting in types. E. g. Iterable[Iterable[Any]] has a level of 2,
        Iterable[Any] of 1. Note that this is assumed for all sub-types in out_arg and in_arg, as this is sufficient for
        the current use case of checking the compatibility of task-to-model and model-to-parser communication flow.
    RETURNS (bool): True if type variables are of the same length and if type variables in out_arg are a subclass
        of (or the same class as) the type variables in in_arg.
    """
    assert hasattr(out_arg, "__args__") and hasattr(in_arg, "__args__")

    out_types, in_types = out_arg, in_arg
    for level in range(nesting_level):
        out_types = (
            out_types.__args__[0] if level < (nesting_level - 1) else out_types.__args__  # type: ignore[attr-defined]
        )
        in_types = (
            in_types.__args__[0] if level < (nesting_level - 1) else in_types.__args__  # type: ignore[attr-defined]
        )
    # Replace Any with object to make issubclass() check work.
    out_types = [arg if arg != Any else object for arg in out_types]
    in_types = [arg if arg != Any else object for arg in in_types]

    if len(out_types) != len(in_types):
        return False

    return all(
        [
            issubclass(out_tv, in_tv) or issubclass(in_tv, out_tv)
            for out_tv, in_tv in zip(out_types, in_types)
        ]
    )


def _extract_model_call_signature(model: PromptExecutorType) -> Dict[str, Any]:
    """Extract call signature from model object.
    model (PromptExecutor): Model object to extract call signature from.
    RETURNS (Dict[str, Any]): Type per argument name.
    """
    if inspect.isfunction(model):
        return typing.get_type_hints(model)

    if not hasattr(model, "__call__"):
        raise ValueError("The object supplied as model must implement `__call__()`.")

    # Assume that __call__() has the necessary type info - except in the case of integration.Model, for which
    # we know this is not the case.
    if not isinstance(model, langchain.LangChain):
        return typing.get_type_hints(model.__call__)

    # If this is an instance of integrations.Model: read type information from .query() instead, only keep
    # information on Iterable args.
    signature = typing.get_type_hints(model.query).items()
    to_ignore: List[str] = []
    for k, v in signature:
        if not (hasattr(v, "__origin__") and issubclass(v.__origin__, Iterable)):
            to_ignore.append(k)

    signature = {
        k: v
        for k, v in typing.get_type_hints(model.query).items()
        # In Python 3.8+ (or 3.6+ if typing_utils is installed) the check for the origin class should be done using
        # typing.get_origin().
        if k not in to_ignore
    }
    assert len(signature) == 2
    assert "return" in signature
    return signature


def supports_sharding(task: Union[NonshardingLLMTask, ShardingLLMTask]) -> bool:
    """Determines task type, as isinstance(instance, Protocol) only checks for method names. This also considers
    argument and return types. Raises an exception if task is neither.
    Note that this is not as thorough as validate_type_consistency() and relies on clues to determine which task type
    a given, type-validated task type is. This doesn't guarantee that a task has valid typing. This method should only
    be in conjunction with validate_type_consistency().
    task (Union[LLMTask, ShardingLLMTask]): Task to check.
    RETURNS (bool): True if task supports sharding, False if not.
    """
    prompt_ret_type = typing.get_type_hints(task.generate_prompts)["return"].__args__[0]
    return (
        hasattr(prompt_ret_type, "_name")
        and prompt_ret_type._name == "Tuple"
        and len(prompt_ret_type.__args__) == 2
    )


def validate_type_consistency(
    task: Union[NonshardingLLMTask, ShardingLLMTask], model: PromptExecutorType
) -> None:
    """Check whether the types of the task and model signatures match.
    task (ShardingLLMTask): Specified task.
    model (PromptExecutor): Specified model.
    """
    # Raises an error or prints a warning if something looks wrong/odd.
    # todo update error messages
    if not isinstance(task, NonshardingLLMTask):
        raise ValueError(
            f"A task needs to adhere to the interface of either 'LLMTask' or 'ShardingLLMTask', but {type(task)} "
            f"doesn't."
        )
    if not hasattr(task, "generate_prompts"):
        raise ValueError(
            "A task needs to have the following method: generate_prompts(self, docs: Iterable[Doc]) -> "
            "Iterable[Tuple[Iterable[Any], Iterable[Doc]]]"
        )
    if not hasattr(task, "parse_responses"):
        raise ValueError(
            "A task needs to have the following method: "
            "parse_responses(self, docs: Iterable[Doc], responses: Iterable[Iterable[Any]]) -> Iterable[Doc]"
        )

    type_hints = {
        "template": typing.get_type_hints(task.generate_prompts),
        "parse": typing.get_type_hints(task.parse_responses),
        "model": _extract_model_call_signature(model),
    }

    parse_in: Optional[Type] = None
    model_in: Optional[Type] = None
    model_out: Optional[Type] = None

    # Validate the 'model' object
    if not (len(type_hints["model"]) == 2 and "return" in type_hints["model"]):
        raise ValueError(
            "The 'model' Callable should have one input argument and one return value."
        )
    for k in type_hints["model"]:
        if k == "return":
            model_out = type_hints["model"][k]
        else:
            model_in = type_hints["model"][k]

    # validate the 'parse' object
    if not (len(type_hints["parse"]) == 3 and "return" in type_hints["parse"]):
        raise ValueError(
            "The 'task.parse_responses()' function should have two input arguments and one return value."
        )
    for k in type_hints["parse"]:
        # find the 'prompt_responses' var without assuming its name
        type_k = type_hints["parse"][k]
        if type_k != typing.Iterable[Doc]:
            parse_in = type_hints["parse"][k]

    template_out = type_hints["template"]["return"]

    # Check that all variables are Iterables.
    for var, msg in (
        (template_out, "`task.generate_prompts()` needs to return an `Iterable`."),
        (
            model_in,
            "The prompts variable in the 'model' needs to be an `Iterable`.",
        ),
        (model_out, "The `model` function needs to return an `Iterable`."),
        (
            parse_in,
            "`responses` in `task.parse_responses()` needs to be an `Iterable`.",
        ),
    ):
        if not (hasattr(var, "_name") and var._name == "Iterable"):
            raise ValueError(msg)

    # Ensure that template/prompt generator output is Iterable of 2-Tuple, the second of which fits doc shards type.
    template_out_type = template_out.__args__[0]
    if (
        hasattr(template_out_type, "_name")
        and template_out_type._name == "Tuple"
        and len(template_out_type.__args__) == 2
    ):
        has_shards = True
        template_out_type = template_out_type.__args__[0]
    else:
        has_shards = False

    # Ensure that the template returns the same type as expected by the model
    assert model_in is not None
    if not _do_args_match(
        template_out_type if has_shards else typing.Iterable[template_out_type],  # type: ignore[valid-type]
        model_in.__args__[0],
        1,
    ):  # type: ignore[arg-type]
        warnings.warn(
            f"First type in value returned from `task.generate_prompts()` (`{template_out_type}`) doesn't match type "
            f"expected by `model` (`{model_in.__args__[0]}`)."
        )

    # Ensure that the parser expects the same type as returned by the model
    if not _do_args_match(model_out, parse_in if has_shards else typing.Iterable[parse_in], 2):  # type: ignore[arg-type,valid-type]
        warnings.warn(
            f"Type returned from `model` (`{model_out}`) doesn't match type expected by "
            f"`task.parse_responses()` (`{parse_in}`)."
        )
