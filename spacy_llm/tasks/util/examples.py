from typing import Dict, List

from pydantic import BaseModel


class SpanExample(BaseModel):
    text: str
    entities: Dict[str, List[str]]


class LemmaExample(BaseModel):
    text: str
    lemmas: List[Dict[str, str]]


class SpanReason(BaseModel):
    text: str
    is_entity: bool
    label: str
    reason: str

    @classmethod
    def from_str(cls, s: str, sep: str = "|"):
        clean_str = s.strip()
        if "." in clean_str:
            clean_str = clean_str.split(".")[1]
        components = [c.strip() for c in clean_str.split(sep)]
        return cls(
            text=components[0],
            is_entity=components[1].lower() == "true",
            label=components[2],
            reason=components[3],
        )

    def __str__(self) -> str:
        return self.to_str()

    def to_str(self) -> str:
        return f"{self.text} | {self.is_entity} | {self.label} | {self.reason}"


class COTSpanExample(BaseModel):
    text: str
    entities: List[SpanReason]
