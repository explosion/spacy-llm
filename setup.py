#!/usr/bin/env python

if __name__ == "__main__":
    from setuptools import setup, find_packages

    setup(
        name="spacy-llm",
        packages=find_packages(),
        entry_points={"spacy_factories": ["llm = spacy_llm.pipeline.llm:make_llm"]},
    )
