import abc
from typing import Literal, Callable, Any, Dict

import requests  # type: ignore
from spacy.pipeline import Pipe
from spacy.tokens import Doc


class RemotePipe(Pipe, abc.ABC):
    """Pipe processing Doc instances remotely."""

    def __init__(
        self,
        endpoint: str,
        doc_to_request_data: Callable[[Doc], Dict[str, Any]],
        set_annotations: Callable[[Doc, Any], Doc],
        method: Literal["GET", "POST"] = "POST",
    ) -> None:
        """
        endpoint (str): URL to address.
        doc_to_request_data (Callable[[Doc], Any]): Callable producing request data from Doc.
        set_annotations (Callable[[Doc, Any], Doc]): Callable setting the API response annotations on the Doc.
        method (Literal["GET", "POST"]): Request method.
        """
        self.endpoint = endpoint
        self.method = method
        self.make_request = doc_to_request_data
        self.set_annotations = set_annotations

    def __call__(self, doc: Doc) -> Doc:
        """Executes request generated from this doc and annotates doc with the results.
        doc (Doc): Doc to generate request from.
        RETURNS (Doc): Annotated doc.
        """
        res = requests.request(
            method=self.method, url=self.endpoint, **self.make_request(doc)
        )
        res.raise_for_status()

        return self.set_annotations(doc, res.json())
