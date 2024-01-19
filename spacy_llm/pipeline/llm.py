import logging
import warnings
from collections import defaultdict
from itertools import tee
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Union, cast

import spacy
from spacy import util
from spacy.language import Language
from spacy.pipeline import Pipe
from spacy.tokens import Doc
from spacy.training import Example
from spacy.ty import InitializableComponent as Initializable
from spacy.vocab import Vocab

from .. import registry  # noqa: F401
from ..compat import TypedDict
from ..ty import Cache, LabeledTask, LLMTask, ModelWithContextLength
from ..ty import PromptExecutorType, ScorableTask, Serializable, supports_sharding
from ..ty import validate_type_consistency

logger = logging.getLogger("spacy_llm")
logger.addHandler(logging.NullHandler())

DEFAULT_MODEL_CONFIG = {
    "@llm_models": "spacy.GPT-3-5.v2",
    "strict": True,
}
DEFAULT_CACHE_CONFIG = {
    "@llm_misc": "spacy.BatchCache.v1",
    "path": None,
    "batch_size": 64,
    "max_batches_in_mem": 4,
}

DEFAULT_SAVE_IO = False
DEFAULT_VALIDATE_TYPES = True


class CacheConfigType(TypedDict):
    path: Optional[Path]
    batch_size: int
    max_batches_in_mem: int


@Language.factory(
    "llm",
    requires=[],
    assigns=[],
    default_config={
        "task": None,
        "model": DEFAULT_MODEL_CONFIG,
        "cache": DEFAULT_CACHE_CONFIG,
        "save_io": DEFAULT_SAVE_IO,
        "validate_types": DEFAULT_VALIDATE_TYPES,
    },
)
def make_llm(
    nlp: Language,
    name: str,
    task: Optional[LLMTask],
    model: PromptExecutorType,
    cache: Cache,
    save_io: bool,
    validate_types: bool,
) -> "LLMWrapper":
    """Construct an LLM component.

    nlp (Language): Pipeline.
    name (str): The component instance name, used to add entries to the
        losses during training.
    task (Optional[LLMTask]): An LLMTask can generate prompts for given docs, and can parse the LLM's responses into
        structured information and set that back on the docs.
    model (Callable[[Iterable[Any]], Iterable[Any]]]): Callable querying the specified LLM API.
    cache (Cache): Cache to use for caching prompts and responses per doc (batch).
    save_io (bool): Whether to save LLM I/O (prompts and responses) in the `Doc._.llm_io` custom extension.
    validate_types (bool): Whether to check if signatures of configured model and task are consistent.
    """
    if task is None:
        raise ValueError(
            "Argument `task` has not been specified, but is required (e. g. {'@llm_tasks': "
            "'spacy.NER.v3'})."
        )
    if validate_types:
        validate_type_consistency(task, model)

    return LLMWrapper(
        name=name,
        task=task,
        save_io=save_io,
        model=model,
        cache=cache,
        vocab=nlp.vocab,
    )


class LLMWrapper(Pipe):
    """Pipeline component for wrapping LLMs."""

    def __init__(
        self,
        name: str = "LLMWrapper",
        *,
        vocab: Vocab,
        task: LLMTask,
        model: PromptExecutorType,
        cache: Cache,
        save_io: bool,
    ) -> None:
        """
        Component managing execution of prompts to LLM APIs and mapping responses back to Doc/Span instances.

        name (str): The component instance name, used to add entries to the
            losses during training.
        vocab (Vocab): Pipeline vocabulary.
        task (LLMTask): An LLMTask can generate prompts for given docs, and can parse the LLM's responses into
            structured information and set that back on the docs.
        model (Callable[[Iterable[Any]], Iterable[Any]]]): Callable querying the specified LLM API.
        cache (Cache): Cache to use for caching prompts and responses per doc (batch).
        save_io (bool): Whether to save LLM I/O (prompts and responses) in the `Doc._.llm_io` custom extension.
        """
        self._name = name
        self._task = task
        self._model = model
        self._cache = cache
        self._save_io = save_io
        self._cache.initialize(vocab, self._task)

        # This is done this way because spaCy's `validate_init_settings` function
        # does not support `**kwargs: Any`.
        # See https://github.com/explosion/spaCy/blob/master/spacy/schemas.py#L111
        if isinstance(self._task, Initializable):
            self.initialize = self._task.initialize

        self._check_sharding()

    def _check_sharding(self):
        context_length: Optional[int] = None
        if isinstance(self._model, ModelWithContextLength):
            context_length = self._model.context_length
        if supports_sharding(self._task) and context_length is None:
            warnings.warn(
                "Task supports sharding, but model does not provide context length. Data won't be sharded, prompt "
                "might exceed the model's context length. Set context length in your config. If you think spacy-llm"
                " should provide the context length for this model automatically, report this to "
                "https://github.com/explosion/spacy-llm/issues."
            )

    @property
    def labels(self) -> Tuple[str, ...]:
        labels: Tuple[str, ...] = tuple()
        if isinstance(self._task, LabeledTask):
            labels = self._task.labels
        return labels

    def add_label(self, label: str, label_definition: Optional[str] = None) -> int:
        if not isinstance(self._task, LabeledTask):
            raise ValueError("The task of this LLM component does not have labels.")
        return self._task.add_label(label, label_definition)

    def clear(self) -> None:
        if not isinstance(self._task, LabeledTask):
            raise ValueError("The task of this LLM component does not have labels.")
        return self._task.clear()

    @property
    def task(self) -> LLMTask:
        return self._task

    def __call__(self, doc: Doc) -> Doc:
        """Apply the LLM wrapper to a Doc instance.

        doc (Doc): The Doc instance to process.
        RETURNS (Doc): The processed Doc.
        """
        docs = self._process_docs([doc])
        assert isinstance(docs[0], Doc)
        return docs[0]

    def score(
        self, examples: Iterable[Example], **kwargs
    ) -> Dict[str, Union[float, Dict[str, float]]]:
        """Score a batch of examples.

        examples (Iterable[Example]): The examples to score.
        RETURNS (Dict[str, Any]): The scores.

        DOCS: https://spacy.io/api/pipe#score
        """
        if isinstance(self._task, ScorableTask):
            return self._task.scorer(examples)
        return {}

    def pipe(self, stream: Iterable[Doc], *, batch_size: int = 128) -> Iterator[Doc]:
        """Apply the LLM prompt to a stream of documents.

        stream (Iterable[Doc]): A stream of documents.
        batch_size (int): The number of documents to buffer.
        YIELDS (Doc): Processed documents in order.
        """
        error_handler = self.get_error_handler()
        for doc_batch in spacy.util.minibatch(stream, batch_size):
            try:
                yield from iter(self._process_docs(doc_batch))
            except Exception as e:
                error_handler(self._name, self, doc_batch, e)

    def _process_docs(self, docs: List[Doc]) -> List[Doc]:
        """Process a batch of docs with the configured LLM model and task.
        If a cache is configured, only sends prompts to model for docs not found in cache.

        docs (List[Doc]): Input batch of docs.
        RETURNS (List[Doc]): Processed batch of docs with task annotations set.
        """
        support_sharding = supports_sharding(self._task)
        is_cached = [doc in self._cache for doc in docs]
        noncached_doc_batch = [doc for i, doc in enumerate(docs) if not is_cached[i]]
        if len(noncached_doc_batch) < len(docs):
            logger.debug(
                "Found %d docs in cache. Processing %d docs not found in cache",
                len(docs) - len(noncached_doc_batch),
                len(noncached_doc_batch),
            )

        # Process uncached docs.
        modified_docs: Iterator[Doc] = iter(())
        if len(noncached_doc_batch) > 0:
            n_iters = 3 if self._save_io else 2
            context_length: Optional[int] = None
            if isinstance(self._model, ModelWithContextLength):
                context_length = self._model.context_length

            # Only pass context length if this is a sharding task.
            prompts_iters = tee(
                self._task.generate_prompts(noncached_doc_batch, context_length)  # type: ignore[call-arg]
                if support_sharding
                else self._task.generate_prompts(noncached_doc_batch),
                n_iters + 1,
            )
            responses_iters = tee(
                self._model(
                    # Ensure that model receives Iterable[Iterable[Any]]. If task doesn't shard, its prompt is wrapped
                    # in a list to conform to the nested structure.
                    (
                        elem[0] if support_sharding else [elem]
                        for elem in prompts_iters[0]
                    )
                ),
                n_iters,
            )

            for prompt_data, response, doc in zip(
                prompts_iters[1], responses_iters[0], noncached_doc_batch
            ):
                logger.debug(
                    "Generated prompt for doc: %s\n%s",
                    doc.text,
                    prompt_data[0] if support_sharding else prompt_data,
                )
                logger.debug("LLM response for doc: %s\n%s", doc.text, response)

            modified_docs = iter(
                self._task.parse_responses(
                    (
                        elem[1] if support_sharding else noncached_doc_batch[i]
                        for i, elem in enumerate(prompts_iters[2])
                    ),
                    responses_iters[1],
                )
            )

        noncached_doc_batch_iter = iter(noncached_doc_batch)
        final_docs: List[Doc] = []
        for i, doc in enumerate(docs):
            if is_cached[i]:
                cached_doc = self._cache[doc]
                assert cached_doc is not None
                cached_doc._context = doc._context
                final_docs.append(cached_doc)
            else:
                doc = next(modified_docs)

                # Merge with doc's prior custom data.
                noncached_doc = next(noncached_doc_batch_iter)
                for extension in dir(noncached_doc._):
                    if not Doc.has_extension(extension):
                        Doc.set_extension(extension, default=None)
                    # Don't overwrite any non-None extension values in new doc.
                    if getattr(doc._, extension) is None:
                        setattr(doc._, extension, getattr(noncached_doc._, extension))
                doc.user_data = {**noncached_doc.user_data, **doc.user_data}
                doc._context = noncached_doc._context

                # Save raw IO (prompt and response), if save_io is True.
                if self._save_io:
                    # Make sure the `llm_io` field is set
                    doc.user_data["llm_io"] = doc.user_data.get(
                        "llm_io", defaultdict(dict)
                    )
                    llm_io = doc.user_data["llm_io"][self._name]
                    next_prompt = next(prompts_iters[-1])
                    if support_sharding:
                        llm_io["prompt"] = [
                            str(shard_prompt) for shard_prompt in next_prompt[0]
                        ]
                        llm_io["response"] = [
                            str(shard_response)
                            for shard_response in next(responses_iters[-1])
                        ]
                    else:
                        llm_io["prompt"] = str(next_prompt)
                        # Models always return nested responses. For non-sharding tasks this will always be a 1-list.
                        llm_io["response"] = str(next(responses_iters[-1])[0])

                self._cache.add(doc)
                final_docs.append(doc)

        return final_docs

    def to_bytes(
        self,
        *,
        exclude: Tuple[str] = cast(Tuple[str], tuple()),
    ) -> bytes:
        """Serialize the LLMWrapper to a bytestring.

        exclude (Tuple): Names of properties to exclude from serialization.
        RETURNS (bytes): The serialized object.
        """

        serialize = {}

        if isinstance(self._task, Serializable):
            serialize["task"] = lambda: self._task.to_bytes(exclude=exclude)  # type: ignore[attr-defined, union-attr]
        if isinstance(self._model, Serializable):
            serialize["model"] = lambda: self._model.to_bytes(exclude=exclude)  # type: ignore[attr-defined]

        return util.to_bytes(serialize, exclude)

    def from_bytes(
        self,
        bytes_data: bytes,
        *,
        exclude: Tuple[str] = cast(Tuple[str], tuple()),
    ) -> "LLMWrapper":
        """Load the LLMWrapper from a bytestring.

        bytes_data (bytes): The data to load.
        exclude (Tuple[str]): Names of properties to exclude from deserialization.
        RETURNS (LLMWrapper): Modified LLMWrapper instance.
        """

        deserialize = {}

        if isinstance(self._task, Serializable):
            deserialize["task"] = lambda b: self._task.from_bytes(b, exclude=exclude)  # type: ignore[attr-defined,union-attr]
        if isinstance(self._model, Serializable):
            deserialize["model"] = lambda b: self._model.from_bytes(b, exclude=exclude)  # type: ignore[attr-defined,union-attr]

        util.from_bytes(bytes_data, deserialize, exclude)
        return self

    def to_disk(
        self, path: Path, *, exclude: Tuple[str] = cast(Tuple[str], tuple())
    ) -> None:
        """Serialize the LLMWrapper to disk.
        path (Path): A path (currently unused).
        exclude (Tuple): Names of properties to exclude from serialization.
        """

        serialize = {}

        if isinstance(self._task, Serializable):
            serialize["task"] = lambda p: self._task.to_disk(p, exclude=exclude)  # type: ignore[attr-defined,union-attr]
        if isinstance(self._model, Serializable):
            serialize["model"] = lambda p: self._model.to_disk(p, exclude=exclude)  # type: ignore[attr-defined,union-attr]

        util.to_disk(path, serialize, exclude)

    def from_disk(
        self, path: Path, *, exclude: Tuple[str] = cast(Tuple[str], tuple())
    ) -> "LLMWrapper":
        """Load the LLMWrapper from disk.
        path (Path): A path (currently unused).
        exclude (Tuple): Names of properties to exclude from deserialization.
        RETURNS (LLMWrapper): Modified LLMWrapper instance.
        """

        serialize = {}

        if isinstance(self._task, Serializable):
            serialize["task"] = lambda p: self._task.from_disk(p, exclude=exclude)  # type: ignore[attr-defined,union-attr]
        if isinstance(self._model, Serializable):
            serialize["model"] = lambda p: self._model.from_disk(p, exclude=exclude)  # type: ignore[attr-defined,union-attr]

        util.from_disk(path, serialize, exclude)
        return self
