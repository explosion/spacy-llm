import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--external",
        action="store_true",
        default=False,
        help="include tests that connects to third-party API",
    )
    parser.addoption(
        "--installed",
        action="store_true",
        default=False,
        help="include tests that require `spacy-llm` to be an installed package",
    )


def pytest_runtest_setup(item):
    def getopt(opt):
        return item.config.getoption(f"--{opt}", False)

    # Integration of boolean flags
    for opt in ["external", "installed"]:
        if opt in item.keywords and not getopt(opt):
            pytest.skip(f"need --{opt} option to run")


def pytest_collection_modifyitems(config, items):
    skip_external = pytest.mark.skip(reason="need --external option to run")
    skip_installed = pytest.mark.skip(reason="need --installed option to run")
    for item in items:
        if "external" in item.keywords and not config.getoption("--external"):
            item.add_marker(skip_external)
        if "installed" in item.keywords and not config.getoption("--installed"):
            item.add_marker(skip_installed)
