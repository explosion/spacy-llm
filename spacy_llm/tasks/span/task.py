import abc
from typing import Any, Callable, Dict, Iterable, List, Optional, Type, TypeVar, Union
from typing import cast

from spacy.tokens import Doc, Span

from ...compat import Literal, Protocol, Self
from ...ty import FewshotExample, ShardMapper, ShardReducer, TaskResponseParser
from ..builtin_task import BuiltinTaskWithLabels
from . import SpanExample
from .examples import SpanCoTExample

SpanTaskContraT = TypeVar("SpanTaskContraT", bound="SpanTask", contravariant=True)


class SpanTaskLabelCheck(Protocol[SpanTaskContraT]):
    """Generic protocol for checking label consistency of SpanTask."""

    def __call__(self, task: SpanTaskContraT) -> Iterable[FewshotExample]:
        ...


class SpanTask(BuiltinTaskWithLabels, abc.ABC):
    """Base class for Span-related tasks, eg NER and SpanCat."""

    def __init__(
        self,
        parse_responses: TaskResponseParser[Self],
        prompt_example_type: Type[Union[SpanExample[Self], SpanCoTExample[Self]]],
        labels: List[str],
        template: str,
        label_definitions: Optional[Dict[str, str]],
        prompt_examples: Optional[
            Union[List[SpanExample[Self]], List[SpanCoTExample[Self]]]
        ],
        shard_mapper: ShardMapper,
        shard_reducer: ShardReducer[Self],
        description: Optional[str],
        normalizer: Optional[Callable[[str], str]],
        alignment_mode: Literal["strict", "contract", "expand"],  # noqa: F821
        case_sensitive_matching: bool,
        allow_overlap: bool,
        single_match: bool,
        check_label_consistency: SpanTaskLabelCheck[Self],
    ):
        super().__init__(
            parse_responses=parse_responses,
            prompt_example_type=prompt_example_type,
            template=template,
            prompt_examples=prompt_examples,
            shard_mapper=shard_mapper,
            shard_reducer=shard_reducer,
            labels=labels,
            label_definitions=label_definitions,
            normalizer=normalizer,
        )

        self._prompt_example_type = cast(
            Type[Union[SpanExample[Self], SpanCoTExample[Self]]],
            self._prompt_example_type,
        )
        self._validate_alignment(alignment_mode)
        self._alignment_mode = alignment_mode
        self._case_sensitive_matching = case_sensitive_matching
        self._allow_overlap = allow_overlap
        self._single_match = single_match
        self._check_label_consistency = check_label_consistency
        self._description = description

        if self._prompt_examples:
            self._prompt_examples = list(self._check_label_consistency(self))

    def _get_prompt_data(
        self, shard: Doc, i_shard: int, i_doc: int, n_shards: int
    ) -> Dict[str, Any]:
        return {
            "description": self._description,
            "labels": list(self._label_dict.values()),
            "label_definitions": self._label_definitions,
            "examples": self._prompt_examples,
            "allow_overlap": self._allow_overlap,
        }

    @staticmethod
    def _validate_alignment(alignment_mode: str):
        """Raises error if specified alignment_mode is not supported.
        alignment_mode (str): Alignment mode to check.
        """
        # ideally, this list should be taken from spaCy, but it's not currently exposed from doc.pyx.
        alignment_modes = ("strict", "contract", "expand")
        if alignment_mode not in alignment_modes:
            raise ValueError(
                f"Unsupported alignment mode '{alignment_mode}'. Supported modes: {', '.join(alignment_modes)}"
            )

    def assign_spans(
        self,
        doc: Doc,
        spans: List[Span],
    ) -> None:
        """Assign spans to the document."""
        raise NotImplementedError()

    def parse_responses(
        self, shards: Iterable[Iterable[Doc]], responses: Iterable[Iterable[str]]
    ) -> Iterable[Doc]:
        shards_teed = self._tee_2d_iterable(shards, 2)

        for shards_for_doc, spans_for_doc in zip(
            shards_teed[0], self._parse_responses(self, shards_teed[1], responses)
        ):
            shards_for_doc = list(shards_for_doc)
            for shard, spans in zip(shards_for_doc, spans_for_doc):
                self.assign_spans(shard, spans)

            yield self._shard_reducer(self, shards_for_doc)  # type: ignore[arg-type]

    @property
    def _cfg_keys(self) -> List[str]:
        return [
            "_label_dict",
            "_template",
            "_label_definitions",
            "_alignment_mode",
            "_case_sensitive_matching",
        ]

    @property
    def alignment_mode(self) -> Literal["strict", "contract", "expand"]:  # noqa: F821
        return self._alignment_mode

    @property
    def case_sensitive_matching(self) -> bool:
        return self._case_sensitive_matching

    @property
    def allow_overlap(self) -> bool:
        return self._allow_overlap

    @property
    def prompt_examples(self) -> Optional[Iterable[FewshotExample]]:
        return self._prompt_examples

    @property
    def prompt_example_type(
        self,
    ) -> Type[Union[SpanExample[Self], SpanCoTExample[Self]]]:
        return self._prompt_example_type

    @property
    def single_match(self) -> bool:
        return self._single_match
