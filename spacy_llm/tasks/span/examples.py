import abc
from typing import Dict, List

from ...ty import FewshotExample


class SpanExample(FewshotExample, abc.ABC):
    text: str
    entities: Dict[str, List[str]]
