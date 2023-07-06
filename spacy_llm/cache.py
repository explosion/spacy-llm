import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import numpy
import srsly  # type: ignore[import]
from spacy.tokens import Doc, DocBin
from spacy.vocab import Vocab

from .registry import registry
from .ty import LLMTask, PromptTemplateProvider


@registry.llm_misc("spacy.BatchCache.v1")
def make_cache(
    path: Optional[Union[str, Path]],
    batch_size: int,
    max_batches_in_mem: int,
):
    return BatchCache(
        path=path, batch_size=batch_size, max_batches_in_mem=max_batches_in_mem
    )


class BatchCache:
    """Utility class handling caching functionality for the `llm` component."""

    _INDEX_NAME: str = "index.jsonl"

    def __init__(
        self,
        path: Optional[Union[str, Path]],
        batch_size: int,
        max_batches_in_mem: int,
    ):
        """Initialize Cache instance. Acts as NoOp if path is None.
        path (Optional[Union[str,Path]]): Cache directory.
        batch_size (int): Number of docs in one batch (file).
        max_batches_in_mem (int): Max. number of batches to hold in memory.
        """
        self._path = Path(path) if path else None

        # Number of Docs in one batch.
        self._batch_size = batch_size
        # Max. number of batches to keep in memory.
        self.max_batches_in_mem = max_batches_in_mem
        self._vocab: Optional[Vocab] = None
        self._prompt_template: Optional[str] = None
        self._prompt_template_checked: bool = False

        # Stores doc hash -> batch hash to allow efficient lookup of available Docs.
        self._doc2batch: Dict[int, int] = {}
        # Hashes of batches loaded into memory.
        self._batch_hashes: List[int] = []
        # Container for currently loaded batch of Docs (batch hash -> doc hash -> Doc).
        self._loaded_docs: Dict[int, Dict[int, Doc]] = {}
        # Queue for processed, not yet persisted docs.
        self._cache_queue: List[Doc] = []
        # Statistics.
        self._stats: Dict[str, int] = {
            "hit": 0,
            "hit_contains": 0,
            "missed": 0,
            "missed_contains": 0,
            "added": 0,
            "persisted": 0,
        }

        self._init_cache_dir()

    def initialize(self, vocab: Vocab, task: LLMTask) -> None:
        """
        Initialize cache with data not available at construction time.
        vocab (Vocab): Vocab object.
        task (LLMTask): Task.
        """
        self._vocab = vocab
        if isinstance(task, PromptTemplateProvider):
            self.prompt_template = task.prompt_template
        else:
            self.prompt_template = ""
            if self._path:
                warnings.warn(
                    "The specified task does not provide its prompt template via `prompt_template()`. This means that "
                    "the cache cannot verify whether all cached documents were generated using the same prompt "
                    "template."
                )

    @property
    def prompt_template(self) -> Optional[str]:
        """Get prompt template.
        RETURNS (Optional[str]): Prompt template string used for docs to cache/cached docs.
        """
        return self._prompt_template

    @prompt_template.setter
    def prompt_template(self, prompt_template: str) -> None:
        """Set prompt template.
        prompt_template (str): Prompt template string used for docs to cache/cached docs.
        """
        self._prompt_template = prompt_template
        if not self._path:
            return

        # Store prompt template as file.
        prompt_template_path = self._path / "prompt_template.txt"
        # If no file exists: store in plain text for easier debugging.
        if not prompt_template_path.exists():
            with open(prompt_template_path, "w") as file:
                file.write(self._prompt_template)

        else:
            # If file exists and prompt template is not equal to new prompt template: raise, as this indicates we are
            # trying to reuse a cache directory with a changed prompt.
            with open(prompt_template_path, "r") as file:
                existing_prompt_template = "".join(file.readlines())
            if hash(existing_prompt_template) != hash(self._prompt_template):
                raise ValueError(
                    f"Prompt template in cache directory ({prompt_template_path}) is not equal with "
                    f"current prompt template. Reset your cache if you are using a new prompt "
                    f"template."
                )

    def _init_cache_dir(self) -> None:
        """Init cache directory with index and prompt template file."""
        if self._path is None:
            return

        if self._path.exists() and not self._path.is_dir():
            raise ValueError("Cache directory exists and is not a directory.")
        self._path.mkdir(parents=True, exist_ok=True)

        # Read from index file, if it exists.
        index_path = self._index_path
        if index_path.exists():
            for rec in srsly.read_jsonl(index_path):
                self._doc2batch = {
                    **self._doc2batch,
                    **{int(k): int(v) for k, v in rec.items()},
                }

    @property
    def _index_path(self) -> Path:
        """Returns full path to index file.
        RETURNS (Path): Full path to index file.
        """
        assert self._path is not None
        return self._path / BatchCache._INDEX_NAME

    def _batch_path(self, batch_id: int) -> Path:
        """Returns full path to batch file.
        batch_id (int): Batch id/hash
        RETURNS (Path): Full path to batch file.
        """
        assert self._path is not None
        return self._path / f"{batch_id}.spacy"

    @staticmethod
    def _doc_id(doc: Doc) -> int:
        """Generate a unique ID for one doc.
        doc (Doc): Doc to generate a unique ID for.
        RETURN (int): Unique ID for this doc.
        """
        return numpy.sum(doc.to_array(["ORTH"]), dtype=numpy.uint64).item()

    @staticmethod
    def _batch_id(doc_ids: Iterable[int]) -> int:
        """Generate a unique ID for a batch, given a set of doc ids
        doc_ids (Iterable[int]): doc ids
        RETURN (int): Unique ID for this batch.
        """
        return numpy.sum(
            numpy.asarray(doc_ids, dtype=numpy.uint64), dtype=numpy.uint64
        ).item()

    def add(self, doc: Doc) -> None:
        """Adds processed doc. Note: Adding a doc does _not_ mean that this doc is immediately persisted to disk. This
        happens only after the specified batch size has been reached or _persist() has been called explicitly.
        doc (Doc): Doc to add to persistence queue.
        """
        if self._path is None:
            return
        if not self._prompt_template_checked and self._prompt_template is None:
            warnings.warn(
                "No prompt template set for Cache object, entailing that consistency of prompt template used "
                "to generate docs cannot be checked. Be mindful to reset your cache whenever you change your "
                "prompt template."
            )
            self._prompt_template_checked = True

        self._cache_queue.append(doc)
        self._stats["added"] += 1
        if len(self._cache_queue) == self._batch_size:
            self._persist()

    def _persist(self) -> None:
        """Persists all processed docs in the queue to disk as one file."""
        assert self._path

        doc_ids = [self._doc_id(doc) for doc in self._cache_queue]
        batch_id = self._batch_id(doc_ids)

        for doc_id in doc_ids:
            self._doc2batch[doc_id] = batch_id

        batch_path = self._batch_path(batch_id)
        DocBin(docs=self._cache_queue, store_user_data=True).to_disk(batch_path)
        srsly.write_jsonl(
            self._index_path,
            lines=[{str(doc_id): str(batch_id)} for doc_id in doc_ids],
            append=True,
            append_new_line=False,
        )
        self._stats["persisted"] += len(self._cache_queue)
        self._cache_queue = []

    def __contains__(self, doc: Doc) -> bool:
        """Checks whether doc has been processed and cached.
        doc (Doc): Doc to check for.
        RETURNS (bool): Whether doc has been processed and cached.
        """
        if self._doc_id(doc) not in self._doc2batch:
            self._stats["missed_contains"] += 1
            return False
        self._stats["hit_contains"] += 1
        return True

    def __getitem__(self, doc: Doc) -> Optional[Doc]:
        """Returns processed doc, if available in cache. Note that if doc is not in the set of currently loaded
        documents, its batch will be loaded (and an older batch potentially discarded from memory).
        If doc is not in cache, None is returned.
        doc (Doc): Unprocessed doc whose processed equivalent should be returned.
        RETURNS (Optional[Doc]): Cached and processed version of doc, if available. Otherwise None.
        """
        doc_id = self._doc_id(doc)
        batch_id = self._doc2batch.get(doc_id, None)

        # Doc is not in cache.
        if not batch_id:
            self._stats["missed"] += 1
            return None
        self._stats["hit"] += 1

        # Doc's batch is currently not loaded.
        if batch_id not in self._loaded_docs:
            if self._path is None:
                raise ValueError(
                    "Cache directory path was not configured. Documents can't be read from cache."
                )
            if self._vocab is None:
                raise ValueError(
                    "Vocab must be set in order to Cache.__get_item__() to work."
                )

            # Discard batch, if maximal number of batches would be exceeded otherwise.
            if len(self._loaded_docs) == self.max_batches_in_mem:
                self._loaded_docs.pop(self._batch_hashes[0])
                self._batch_hashes = self._batch_hashes[1:]

            # Load target batch.
            self._batch_hashes.append(batch_id)
            self._loaded_docs[batch_id] = {
                self._doc_id(proc_doc): proc_doc
                for proc_doc in DocBin()
                .from_disk(self._batch_path(batch_id))
                .get_docs(self._vocab)
            }

        return self._loaded_docs[batch_id][doc_id]
