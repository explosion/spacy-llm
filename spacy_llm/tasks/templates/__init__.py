from pathlib import Path

TEMPLATE_DIR = Path(__file__).parent


def read_template(name: str) -> str:
    """Read a template"""

    path = TEMPLATE_DIR / f"{name}.jinja"

    if not path.exists():
        raise ValueError(f"{name} is not a valid template.")

    return path.read_text()
