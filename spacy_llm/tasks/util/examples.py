from typing import Dict, List

from pydantic import BaseModel


class LemmaExample(BaseModel):
    text: str
    lemmas: List[Dict[str, str]]
