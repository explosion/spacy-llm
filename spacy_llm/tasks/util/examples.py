from typing import Dict, List

from pydantic import BaseModel


class SpanExample(BaseModel):
    text: str
    entities: Dict[str, List[str]]
