import inspect
import typing
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, Union
from typing import cast

from spacy.tokens import Doc
from spacy.training.example import Example
from spacy.vocab import Vocab

from .compat import Protocol, runtime_checkable
from .models import langchain

_Prompt = Any
_Response = Any

PromptExecutor = Callable[[Iterable[_Prompt]], Iterable[_Response]]
ExamplesConfigType = Union[
    Iterable[Dict[str, Any]], Callable[[], Iterable[Dict[str, Any]]], None
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
class Scorable(Protocol):
    def scorer(
        self,
        examples: Iterable[Example],
    ) -> Dict[str, Any]:
        """Scores performance on examples."""
        ...


@runtime_checkable
class LLMTask(Protocol):
    def generate_prompts(self, docs: Iterable[Doc]) -> Iterable[_Prompt]:
        ...

    def parse_responses(
        self, docs: Iterable[Doc], responses: Iterable[_Response]
    ) -> Iterable[Doc]:
        ...


@runtime_checkable
class PromptTemplateProvider(Protocol):
    @property
    def prompt_template(self) -> str:
        ...


@runtime_checkable
class Labeled(Protocol):
    @property
    def labels(self) -> Tuple[str, ...]:
        ...


@runtime_checkable
class Cache(Protocol):
    """Defines minimal set of operations a cache implementiation needs to support."""

    def initialize(self, vocab: Vocab, task: LLMTask) -> None:
        """
        Initialize cache with data not available at construction time.
        vocab (Vocab): Vocab object.
        task (LLMTask): Task.
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


def _do_args_match(out_arg: Iterable, in_arg: Iterable) -> bool:
    """Compares argument type of Iterables for compatibility.
    in_arg (Iterable): Input argument.
    out_arg (Iterable): Output argument.
    RETURNS (bool): True if type variables are of the same length and if type variables in out_arg are a subclass
        of (or the same class as) the type variables in in_arg.
    """
    assert hasattr(out_arg, "__args__") and hasattr(in_arg, "__args__")
    # Replace Any with object to make issubclass() check work.
    out_type_vars = [arg if arg != Any else object for arg in out_arg.__args__]
    in_type_vars = [arg if arg != Any else object for arg in in_arg.__args__]

    if len(out_type_vars) != len(in_type_vars):
        return False

    return all(
        [
            issubclass(out_tv, in_tv) or issubclass(in_tv, out_tv)
            for out_tv, in_tv in zip(out_type_vars, in_type_vars)
        ]
    )


def _extract_model_call_signature(model: PromptExecutor) -> Dict[str, Any]:
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


def validate_type_consistency(task: LLMTask, model: PromptExecutor) -> None:
    """Check whether the types of the task and model signatures match.
    task (LLMTask): Specified task.
    backend (PromptExecutor): Specified model.
    """
    # Raises an error or prints a warning if something looks wrong/odd.
    if not isinstance(task, LLMTask):
        raise ValueError(
            f"A task needs to be of type 'LLMTask' but found {type(task)} instead"
        )
    if not hasattr(task, "generate_prompts"):
        raise ValueError(
            "A task needs to have the following method: generate_prompts(self, docs: Iterable[Doc]) -> Iterable[Any]"
        )
    if not hasattr(task, "parse_responses"):
        raise ValueError(
            "A task needs to have the following method: "
            "parse_responses(self, docs: Iterable[Doc], responses: Iterable[Any]) -> Iterable[Doc]"
        )

    type_hints = {
        "template": typing.get_type_hints(task.generate_prompts),
        "parse": typing.get_type_hints(task.parse_responses),
        "model": _extract_model_call_signature(model),
    }

    parse_input: Optional[Type] = None
    model_input: Optional[Type] = None
    model_output: Optional[Type] = None

    # Validate the 'model' object
    if not (len(type_hints["model"]) == 2 and "return" in type_hints["model"]):
        raise ValueError(
            "The 'model' Callable should have one input argument and one return value."
        )
    for k in type_hints["model"]:
        if k == "return":
            model_output = type_hints["model"][k]
        else:
            model_input = type_hints["model"][k]

    # validate the 'parse' object
    if not (len(type_hints["parse"]) == 3 and "return" in type_hints["parse"]):
        raise ValueError(
            "The 'task.parse_responses()' function should have two input arguments and one return value."
        )
    for k in type_hints["parse"]:
        # find the 'prompt_responses' var without assuming its name
        type_k = type_hints["parse"][k]
        if type_k != typing.Iterable[Doc]:
            parse_input = type_hints["parse"][k]

    template_output = type_hints["template"]["return"]

    # Check that all variables are Iterables.
    for var, msg in (
        (template_output, "`task.generate_prompts()` needs to return an `Iterable`."),
        (
            model_input,
            "The prompts variable in the 'model' needs to be an `Iterable`.",
        ),
        (model_output, "The `model` function needs to return an `Iterable`."),
        (
            parse_input,
            "`responses` in `task.parse_responses()` needs to be an `Iterable`.",
        ),
    ):
        if not var != Iterable:
            raise ValueError(msg)

    # Ensure that the template returns the same type as expected by the model
    if not _do_args_match(template_output, model_input):  # type: ignore[arg-type]
        warnings.warn(
            f"Type returned from `task.generate_prompts()` (`{template_output}`) doesn't match type expected by "
            f"`model` (`{model_input}`)."
        )

    # Ensure that the parser expects the same type as returned by the model
    if not _do_args_match(model_output, parse_input):  # type: ignore[arg-type]
        warnings.warn(
            f"Type returned from `model` (`{model_output}`) doesn't match type expected by "
            f"`task.parse_responses()` (`{parse_input}`)."
        )
