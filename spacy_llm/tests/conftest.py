import os
from typing import Iterable

import pytest

from spacy_llm.registry.util import registry


def pytest_addoption(parser):
    parser.addoption(
        "--external",
        action="store_true",
        default=bool(int(os.environ.get("TEST_EXTERNAL", 0))),
        help="include tests that connects to third-party API",
    )
    parser.addoption(
        "--gpu",
        action="store_true",
        default=bool(int(os.environ.get("TEST_GPU", 0))),
        help="include tests that use a GPU",
    )


def pytest_runtest_setup(item):
    def getopt(opt):
        return item.config.getoption(f"--{opt}", False)

    # Integration of boolean flags
    for opt in ["external", "gpu"]:
        if opt in item.keywords and not getopt(opt):
            pytest.skip(f"need --{opt} option to run")


def pytest_collection_modifyitems(config, items):
    types = ("external", "gpu")
    skip_marks = [pytest.mark.skip(reason=f"need --{t} option to run") for t in types]
    for item in items:
        for t, sm in zip(types, skip_marks):
            if (not config.getoption(f"--{t}")) and (t in item.keywords):
                item.add_marker(sm)


@registry.llm_models("test.NoOpModel.v1")
def noop_factory(output: str = ""):
    def noop(prompts: Iterable[str]) -> Iterable[str]:
        return [output] * len(list(prompts))

    return noop
