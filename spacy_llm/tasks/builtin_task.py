import abc
from itertools import tee
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, cast

import jinja2
import jinja2.sandbox
import srsly
from spacy import Errors, Language, util
from spacy.tokens import Doc
from spacy.training import Example

from ..compat import Self
from ..registry import lowercase_normalizer
from ..ty import FewshotExample, ShardMapper, ShardReducer, TaskResponseParser


class BuiltinTask(abc.ABC):
    """Abstract base task implementing interfaces and/or functionality expected from all built-in tasks:
        - working prompt template strings
        - swappable response parsers
        - swappable prompt example type
        - integration of fewshot example into the fully rendered prompt
        - initializable (in line with other spaCy components)
        - (de-)serialization

    On the relation of BuiltinTask to ShardingLLMTask: the latter specifies the minimal contract a task implementation
    has to fulfill, whereas a BuiltinTask requires (and offers) functionality beyond that. The rationale behind that is
    that built-in tasks should provide as smooth a usage experience as possible while still making it as easy as
    possible for users to write their own, custom tasks.
    """

    def __init__(
        self,
        parse_responses: TaskResponseParser,
        prompt_example_type: Type[FewshotExample[Self]],
        template: str,
        prompt_examples: Optional[List[FewshotExample[Self]]],
        shard_mapper: ShardMapper,
        shard_reducer: ShardReducer[Self],
    ):
        """Initializes task.
        parse_responses (TaskResponseParser[Self]): Callable for parsing LLM responses for this task.
        prompt_example_type (Type[FewshotExample[Self]): Type to use for fewshot examples.
        template (str): Prompt template passed to the model.
        prompt_examples (Optional[List[FewshotExample[Self]]]): Optional list of few-shot examples to include in prompts.
        shard_mapper (ShardMapper): Maps docs to shards if they don't fit into the model context.
        shard_reducer (ShardReducer[Self]): Reduces doc shards back into one doc instance.
        """
        self._parse_responses = parse_responses
        self._prompt_examples = prompt_examples or []
        self._template = template
        self._prompt_example_type = prompt_example_type
        self._shard_mapper = shard_mapper
        self._shard_reducer = shard_reducer

    def generate_prompts(
        self, docs: Iterable[Doc], context_length: Optional[int] = None
    ) -> Iterable[Tuple[Iterable[Any], Iterable[Doc]]]:
        """Generate prompts from docs.
        docs (Iterable[Doc]): Docs to generate prompts from.
        ontext_length (int): Context length for model this task is executed with. Needed for sharding and fusing docs,
            if the corresponding prompts exceed the context length. If None, context length is assumed to be infinite.
        RETURNS (Iterable[Tuple[Iterable[Any], Iterable[Doc]]]): Iterable with one to n prompts per doc (multiple
            prompts in case of multiple shards) and the corresponding shards. The relationship between shard and prompt
            is 1:1.
        """
        environment = jinja2.sandbox.SandboxedEnvironment()
        _template = environment.from_string(self._template)

        def render_template(shard: Doc, i_shard: int, i_doc: int, n_shards: int) -> str:
            """Renders template for a given doc (shard).
            shard (Doc): Doc shard. Note that if the prompt is small enough to fit within the model's context window,
                there will only be one shard, which is identical to the original doc.
            i_shard (int): Shard index (w.r.t. shard's Doc instance).
            i_doc (int): Doc index.
            n_shards (int): Total number of shards.
            RETURNS (str): Rendered template.
            """
            return _template.render(
                text=shard.text,
                prompt_examples=self._prompt_examples,
                **self._get_prompt_data(shard, i_shard, i_doc, n_shards),
            )

        for _i_doc, _doc in enumerate(self._preprocess_docs_for_prompt(docs)):
            # If no context length provided (e. g. because models don't provide it): don't shard.
            shards = (
                self._shard_mapper(_doc, _i_doc, context_length, render_template)
                if context_length is not None
                else [_doc]
            )
            shards = list(shards)
            yield [
                render_template(_shard, _i_shard, _i_doc, len(shards))
                for _i_shard, _shard in enumerate(shards)
            ], shards

    def _get_prompt_data(
        self, shard: Doc, i_shard: int, i_doc: int, n_shards: int
    ) -> Dict[str, Any]:
        """Returns data injected into prompt template. No-op if not overridden by inheriting task class. The data
        returned by this might be static (i. e. the same for all doc shards) or dynamic (contingent on the doc shard).
        shard (Doc): Doc (shard) for which prompt data should be fetched.
        i_shard (int): Shard index (w.r.t. shard's Doc instance).
        i_doc (int): Doc index.
        n_shards (int): Total number of shards.
        RETURNS (Dict[str, Any]): Data injected into prompt template.
        """
        return {}

    def _preprocess_docs_for_prompt(self, docs: Iterable[Doc]) -> Iterable[Doc]:
        """Preprocesses docs before injection into prompt template. No-op if not overridden by inheriting task class.
        docs (Iterable[Doc]): Docs to generate prompts from.
        RETURNS (Iterable[Doc]): Preprocessed docs.
        """
        return docs

    @abc.abstractmethod
    def parse_responses(
        self, shards: Iterable[Iterable[Doc]], responses: Iterable[Iterable[Any]]
    ) -> Iterable[Doc]:
        """
        Parses LLM responses.
        shards (Iterable[Iterable[Doc]]): Doc shards to map responses into.
        responses ([Iterable[Iterable[Any]]]): LLM responses per doc.
        RETURNS (Iterable[Doc]]): Updated docs.
        """

    def _initialize(
        self,
        get_examples: Callable[[], Iterable["Example"]],
        nlp: Language,
        n_prompt_examples: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initializes prompt examples from Doc examples.
        get_examples (Callable[[], Iterable["Example"]]): Callable that provides examples
            for initialization.
        nlp (Language): Language instance.
        n_prompt_examples (int): How many prompt examples to infer from the provided Example objects.
            0 by default. Takes all examples if set to -1.
        """
        for eg in get_examples():
            if n_prompt_examples < 0 or len(self._prompt_examples) < n_prompt_examples:
                prompt_example = self._prompt_example_type.generate(eg, self)  # type: ignore[arg-type]
                if prompt_example:
                    self._prompt_examples.append(prompt_example)

    def get_cfg(self) -> Dict[str, Any]:
        """Serialize the task's configuration attributes."""
        cfg = {key: getattr(self, key) for key in self._cfg_keys}
        return cfg

    def set_cfg(self, cfg: Dict[str, Any]) -> None:
        """Deserialize the task's configuration attributes.
        cfg (Dict[str, Any]): dictionary containing configuration attributes.
        """
        for key, value in cfg.items():
            setattr(self, key, value)

    def _get_prompt_examples(self) -> List[Dict[str, Any]]:
        """Serialize examples."""
        examples = [eg.dict() for eg in self._prompt_examples]
        return examples

    def _set_prompt_examples(self, examples: List[Dict[str, Any]]) -> None:
        """Set prompt examples.
        examples (List[Dict[str, Any]]): prompt examples.
        """
        self._prompt_examples = [
            self._prompt_example_type.parse_obj(eg) for eg in examples
        ]

    def to_bytes(
        self,
        *,
        exclude: Tuple[str] = cast(Tuple[str], tuple()),
    ) -> bytes:
        """Serialize the BuiltinTask to a bytestring.

        exclude (Tuple): Names of properties to exclude from serialization.
        RETURNS (bytes): The serialized object.
        """
        serialize = {
            "cfg": lambda: srsly.json_dumps(self.get_cfg()),
            "prompt_examples": lambda: srsly.msgpack_dumps(self._get_prompt_examples()),
        }

        return util.to_bytes(serialize, exclude)

    def from_bytes(
        self,
        bytes_data: bytes,
        *,
        exclude: Tuple[str] = cast(Tuple[str], tuple()),
    ) -> "BuiltinTask":
        """Load the BuiltinTask from a bytestring.

        bytes_data (bytes): The data to load.
        exclude (Tuple[str]): Names of properties to exclude from deserialization.
        RETURNS (BuiltinTask): Modified BuiltinTask instance.
        """
        deserialize = {
            "cfg": lambda b: self.set_cfg(srsly.json_loads(b)),
            "prompt_examples": lambda b: self._set_prompt_examples(
                srsly.msgpack_loads(b)
            ),
        }

        util.from_bytes(bytes_data, deserialize, exclude)
        return self

    def to_disk(
        self,
        path: Path,
        *,
        exclude: Tuple[str] = cast(Tuple[str], tuple()),
    ) -> None:
        """Serialize the task to disk.

        path (Path): A path (currently unused).
        exclude (Tuple): Names of properties to exclude from serialization.
        """
        serialize = {
            "cfg": lambda p: srsly.write_json(p, self.get_cfg()),
            "prompt_examples": lambda p: srsly.write_msgpack(
                p, self._get_prompt_examples()
            ),
        }

        util.to_disk(path, serialize, exclude)

    def from_disk(
        self,
        path: Path,
        *,
        exclude: Tuple[str] = cast(Tuple[str], tuple()),
    ) -> "BuiltinTask":
        """Deserialize the task from disk.

        path (Path): A path (currently unused).
        exclude (Tuple): Names of properties to exclude from serialization.
        RETURNS (BuiltinTask): The BuiltinTask instance.
        """

        deserialize = {
            "cfg": lambda p: self.set_cfg(srsly.read_json(p)),
            "prompt_examples": lambda p: self._set_prompt_examples(
                srsly.read_msgpack(p)
            ),
        }

        util.from_disk(path, deserialize, exclude)
        return self

    @property
    def prompt_template(self) -> str:
        return self._template

    @property
    @abc.abstractmethod
    def _cfg_keys(self) -> List[str]:
        """A list of configuration attributes to serialize."""
        pass

    @classmethod
    def _check_extension(cls, extension: str) -> None:
        """Add extension if need be.
        extension (str): Extension to check/add.
        """
        if not Doc.has_extension(extension):
            Doc.set_extension(extension, default=[])

    @staticmethod
    def _tee_2d_iterable(
        data: Iterable[Iterable[Any]], n: int
    ) -> Tuple[Iterable[List[Doc]], ...]:
        """Tees two-dimensional Iterable. As Iterables in the nested iterables get consumed with the first access, we
        need to materialize them - this is done by converting them to a list.
        data (Iterable[Iterable[Any]]): Data to tee.
        n (int): Number of tees to return.
        RETURNS (Tuple[Iterable[List[Doc]], ...]): n-sized tuple of Iterables with inner Iterables converted to Lists.
        """
        return tee((list(inner_data) for inner_data in data), n)


class BuiltinTaskWithLabels(BuiltinTask, abc.ABC):
    """Built-in tasks with labels."""

    def __init__(
        self,
        parse_responses: TaskResponseParser,
        prompt_example_type: Type[FewshotExample[Self]],
        template: str,
        prompt_examples: Optional[List[FewshotExample[Self]]],
        shard_mapper: ShardMapper,
        shard_reducer: ShardReducer[Self],
        labels: List[str],
        label_definitions: Optional[Dict[str, str]],
        normalizer: Optional[Callable[[str], str]],
    ):
        """Built-in task with labels.

        parse_responses (TaskResponseParser[Self]): Callable for parsing LLM responses for this task.
        prompt_example_type (Type[FewshotExample[Self]): Type to use for fewshot examples.
        template (str): Prompt template passed to the model.
        prompt_examples (Optional[List[FewshotExample[Self]]]): Optional list of few-shot examples to include in prompts.
        shard_mapper (ShardMapper): Maps docs to shards if they don't fit into the model context.
        shard_reducer (ShardReducer[Self]): Reduces doc shards back into one doc instance.
        labels (List[str]): List of labels to pass to the template.
            Leave empty to (optionally) populate it at initialization time.
        label_definitions (Optional[Dict[str, str]]): Map of label -> description
            of the label to help the language model output the entities wanted.
            It is usually easier to provide these definitions rather than
            full examples, although both can be provided.
        normalizer (Optional[Callable[[str], str]]): optional normalizer function.
        """
        super().__init__(
            parse_responses=parse_responses,
            prompt_example_type=prompt_example_type,
            template=template,
            prompt_examples=prompt_examples,
            shard_mapper=shard_mapper,
            shard_reducer=shard_reducer,
        )
        self._normalizer = normalizer if normalizer else lowercase_normalizer()
        self._label_dict = {
            self._normalizer(label): label for label in sorted(set(labels))
        }
        self._label_definitions = label_definitions

    def _initialize(  # type: ignore[override]
        self,
        get_examples: Callable[[], Iterable["Example"]],
        nlp: Language,
        labels: List[str] = [],
        n_prompt_examples: int = 0,
        **kwargs,
    ) -> None:
        """Supports initialization of tasks with labels by auto-discovering labels and returning the derived few-shot
        examples and label dict.

        Labels can be set through, by order of precedence:

        - the `[initialize]` section of the pipeline configuration
        - the `labels` argument supplied to the task factory
        - the labels found in the examples

        get_examples (Callable[[], Iterable["Example"]]): Callable that provides examples
            for initialization.
        nlp (Language): Language instance.
        labels (List[str]): Optional list of labels.
        n_prompt_examples (int): How many prompt examples to infer from the Example objects.
            0 by default. Takes all examples if set to -1.
        """
        if not labels:
            labels = list(self._label_dict.values())
        infer_labels = not labels

        if infer_labels:
            labels = []

        for eg in get_examples():
            if infer_labels:
                labels.extend(self._extract_labels_from_example(eg))
            if n_prompt_examples < 0 or len(self._prompt_examples) < n_prompt_examples:
                prompt_example = self._prompt_example_type.generate(eg, self)  # type: ignore[arg-type]
                if prompt_example:
                    self._prompt_examples.append(prompt_example)

        self._label_dict = {
            self._normalizer(label): label for label in sorted(set(labels))
        }

    @abc.abstractmethod
    def _extract_labels_from_example(self, example: Example) -> List[str]:
        """Extracts labels from Example instance.
        example (Example): Example to extract labels from.
        RETURNS (List[str]): Labels extracted from Example instance.
        """

    @property
    def labels(self) -> Tuple[str, ...]:
        return tuple(self._label_dict.values())

    def add_label(self, label: str, label_definition: Optional[str] = None) -> int:
        """Add a label to the task"""
        if not isinstance(label, str):
            raise ValueError(Errors.E187)
        if label in self.labels:
            return 0
        self._label_dict[self._normalizer(label)] = label
        if label_definition is None:
            return 1
        if self._label_definitions is None:
            self._label_definitions = {}
        self._label_definitions[label] = label_definition
        return 1

    def clear(self) -> None:
        """Reset all labels."""
        self._label_dict = {}
        self._label_definitions = None

    @property
    def normalizer(self) -> Callable[[str], str]:
        return self._normalizer

    @property
    def label_dict(self) -> Dict[str, str]:
        return self._label_dict

    @property
    def label_definitions(self) -> Optional[Dict[str, str]]:
        return self._label_definitions
