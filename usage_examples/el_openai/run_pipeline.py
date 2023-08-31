import os
from pathlib import Path
from typing import Optional

import spacy.util
import typer
from spacy.tokens import Doc
from wasabi import msg

from spacy_llm.tests.tasks.test_entity_linker import build_el_pipeline
from spacy_llm.util import assemble

Arg = typer.Argument
Opt = typer.Option


def run_pipeline(
    # fmt: off
    text: str = Arg("", help="Text to perform text categorization on."),
    config_path: Path = Arg(..., help="Path to the configuration file to use."),
    examples_path: Optional[Path] = Arg(None, help="Path to the examples file to use (few-shot only)."),
    verbose: bool = Opt(False, "--verbose", "-v", help="Show extra information."),
    # fmt: on
) -> Doc:
    if not os.getenv("OPENAI_API_KEY", None):
        msg.fail(
            "OPENAI_API_KEY env variable was not found. "
            "Set it by running 'export OPENAI_API_KEY=...' and try again.",
            exits=1,
        )

    with spacy.util.make_tempdir() as tmpdir:
        # Construct toy KB.
        build_el_pipeline(tmpdir, tmpdir / "desc.csv")
        paths = {
            "paths.el_nlp": str(tmpdir),
            "paths.el_desc": str(tmpdir / "desc.csv"),
        }
        msg.text(f"Loading config from {config_path}", show=verbose)

        nlp = assemble(
            config_path,
            overrides={**paths}
            if examples_path is None
            else {**paths, "paths.examples": str(examples_path)},
        )

        doc = nlp(text)

    msg.text(f"Text: {doc.text}")
    msg.text(f"Entities: {[(ent.text, ent.label_, ent.kb_id_) for ent in doc.ents]}")

    return doc


if __name__ == "__main__":
    typer.run(run_pipeline)
