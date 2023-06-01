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


def pytest_runtest_setup(item):
    def getopt(opt):
        return item.config.getoption(f"--{opt}", False)

    # Integration of boolean flags
    for opt in ["external"]:
        if opt in item.keywords and not getopt(opt):
            pytest.skip(f"need --{opt} option to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--external"):
        return

    skip_external = pytest.mark.skip(reason="need --external option to run")
    for item in items:
        if "external" in item.keywords:
            item.add_marker(skip_external)


@registry.llm_backends("test.NoOpBackend.v1")
def noop_factory(output: str = ""):
    def noop(prompts: Iterable[str]) -> Iterable[str]:
        return [output] * len(list(prompts))

    return noop
