import os
import typer

from pathlib import Path
from spacy_llm.util import assemble
from typing import Optional
from wasabi import msg

Arg = typer.Argument
Opt = typer.Option


def run_pipeline(
    # fmt: off
    text: str = Arg("", help="Text to perform text categorization on."),
    config_path: Path = Arg(..., help="Path to the configuration file to use."),
    examples_path: Optional[Path] = Arg(None, help="Path to the examples file to use (few-shot only)."),
    verbose: bool = Opt(False, "--verbose", "-v", help="Show extra information."),
    # fmt: on
):
    if not os.getenv("OPENAI_API_KEY", None):
        msg.fail(
            "OPENAI_API_KEY env variable was not found. "
            "Set it by running 'export OPENAI_API_KEY=...' and try again.",
            exits=1,
        )

    msg.text(f"Loading config from {config_path}", show=verbose)
    nlp = assemble(
        config_path,
        overrides={}
        if examples_path is None
        else {"paths.examples": str(examples_path)},
    )

    doc = nlp(text)

    msg.text(f"Text: {doc.text}")
    msg.text(f"Predicates: {[p['text'] for p in doc._.predicates]}")

    msg.text("Relations:")
    for r in doc._.relations:
        msg.text(f"  - {r['predicate']['text']} [{r['label']}] {r['role']['text']}")


if __name__ == "__main__":
    typer.run(run_pipeline)
