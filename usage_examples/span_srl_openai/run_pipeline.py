import os
import typer

from pathlib import Path
from spacy_llm.util import assemble
from spacy_llm.tasks.srl_task import SRLExample
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

    doc_srl = SRLExample(
        text=doc.text, predicates=doc._.predicates, relations=doc._.relations
    )

    msg.text(f"Text: {doc_srl.text}")
    msg.text(f"SRL Output:\n{str(doc_srl)}\n")


if __name__ == "__main__":
    typer.run(run_pipeline)
