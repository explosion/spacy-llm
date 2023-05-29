import pytest
from spacy import registry

FUNCTIONS = [
    ("misc", "spacy.FewShotReader.v1"),
    ("misc", "spacy.FileReader.v1"),
]


@pytest.mark.parametrize("reg_name,func_name", FUNCTIONS)
def test_registry(reg_name, func_name):
    assert registry.has(reg_name, func_name)
    assert registry.get(reg_name, func_name)
