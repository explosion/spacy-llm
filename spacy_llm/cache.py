import os
from pathlib import Path
from typing import Dict, Union, Optional, Iterable, List

import srsly
from spacy import Vocab
from spacy.tokens import Doc, DocBin


class Cache:
    """Utility class handling caching functionality for the `llm` component."""

    _index_name: str = "index.jsonl"

    def __init__(
        self, path: Union[str, Path], batch_size: int, max_n_batches: int, vocab: Vocab
    ):
        """Initialize Cache instance.
        path (Path): Cache directory.
        batch_size (int): Number of docs in one batch (file).
        max_n_batches (int): Max. number of batches to hold in memory.
        vocab (Vocab): Vocab object.
        """
        self._path = Path(path)
        # Number of Docs in one batch.
        self._batch_size = batch_size
        # Max. number of batches to keep in memory.
        self._max_n_batches = max_n_batches
        self._vocab = vocab

        # Stores doc hash -> batch hash to allow efficient lookup of available Docs.
        self._index: Dict[int, int] = {}
        # Hashes of batches loaded into memory.
        self._batch_hashes: List[int] = []
        # Container for currently loaded batch of Docs (batch hash -> doc hash -> Doc).
        self._cached_docs: Dict[int, Dict[int, Doc]] = {}
        # Queue for processed, not yet persisted docs.
        self._docs_to_be_cached: List[Doc] = []

        self._init_cache_index()

    def _init_cache_index(self) -> None:
        """Init cache index and directory."""
        if self._path is None:
            self._index = {}
            return

        if self._path.exists() and not os.path.isdir(self._path):
            raise ValueError("Cache directory exists and is not a directory.")
        self._path.mkdir(parents=True, exist_ok=True)

        index_path = self._path / Cache._index_name
        if index_path.exists():
            self._index = {
                rec["doc_hash"]: rec["batch_hash"]
                for rec in srsly.read_jsonl(index_path)
            }

    @staticmethod
    def _batch_id(docs: Iterable[Doc]) -> int:
        """Hash docs.
        docs (Iterable[Doc]): Docs to hash.
        RETURN (int): Hash for this collection of docs.
        """
        return sum(sum(token.orth for token in doc) for doc in docs)

    def add(self, doc: Doc) -> None:
        """Adds processed doc. Note: Adding a doc does _not_ mean that this doc is immediately persisted to disk. This
        happens only after the specified batch size has been reached or persist() has been called explicitly.
        doc (Doc): Doc to add to persistence queue.
        """
        self._docs_to_be_cached.append(doc)
        if len(self._docs_to_be_cached) == self._batch_size:
            self._persist()

    def _persist(self) -> None:
        """Persists all processed docs in the queue to disk as one file."""
        batch_hash = self._batch_id(self._docs_to_be_cached)
        DocBin(docs=self._docs_to_be_cached, store_user_data=True).to_disk(
            self._path / f"{batch_hash}.spacy"
        )
        srsly.write_jsonl(
            self._path / Cache._index_name,
            lines=[{hash(doc): batch_hash for doc in self._docs_to_be_cached}],
            append=True,
        )
        self._docs_to_be_cached = []

    def __contains__(self, doc: Doc) -> bool:
        """Checks whether doc has been processed and cached.
        doc (Doc): Doc to check for.
        RETURNS (bool): Whether doc has been processed and cached.
        """
        return self._batch_id([doc]) in self._index

    def __getitem__(self, doc: Doc) -> Optional[Doc]:
        """Returns processed doc, if available in cache. Note that if doc is not in the set of currently loaded
        documents, its batch will be loaded (and an older batch potentially discarded from memory).
        If doc is not in cache, None is returned.
        doc (Doc): Unprocessed doc whose processed equivalent should be returned.
        RETURNS (Optional[Doc]): Cached and processed version of doc, if available. Otherwise None.
        """
        doc_hash = self._batch_id([doc])
        batch_hash = self._index.get(doc_hash, None)

        # Doc is not in cache.
        if not batch_hash:
            return None

        # Doc's batch is currently not loaded.
        if batch_hash not in self._cached_docs:
            # Discard batch, if necessary.
            if len(self._cached_docs) == self._max_n_batches:
                self._cached_docs.pop(self._batch_hashes[0])
                self._batch_hashes = self._batch_hashes[1:]

            # Load target batch.
            self._batch_hashes.append(batch_hash)
            self._cached_docs[batch_hash] = {
                self._batch_id([proc_doc]): proc_doc
                for proc_doc in DocBin()
                .from_disk(self._path / f"{batch_hash}.spacy")
                .get_docs(self._vocab)
            }

        return self._cached_docs[batch_hash][doc_hash]
