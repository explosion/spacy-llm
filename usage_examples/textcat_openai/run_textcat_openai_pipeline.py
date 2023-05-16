import os
from pathlib import Path

import typer
from spacy import util
from wasabi import msg


Arg = typer.Argument
Opt = typer.Option


def run_pipeline(
    # fmt: off
    text: str = Arg("", help="Text to perform text categorization on."),
    config_path: Path = Arg(..., help="Path to the configuration file to use."),
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
    config = util.load_config(config_path)
    # Reload config with dynamic path for examples, if available in config.
    if "examples" in config.get("paths", {}):
        config = util.load_config(
            config_path,
            overrides={
                "paths.examples": str(Path(__file__).parent / "textcat_examples.jsonl")
            },
        )

    nlp = util.load_model_from_config(config, auto_fill=True)
    doc = nlp(text)

    msg.text(f"Text: {doc.text}")
    msg.text(f"Categories: {doc.cats}")


if __name__ == "__main__":
    typer.run(run_pipeline)
