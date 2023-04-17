from typing import Callable, Optional, Iterable

import minichain
import spacy
from spacy import registry
from spacy.tokens import Doc

# Create new registry.
spacy.registry.create("llm", entry_points=True)


@registry.llm("spacy.DummyTemplate.v1")
def dummy_template() -> Callable[[Iterable[Doc]], Iterable[str]]:
    """Returns Callable injecting Doc data into a prompt template and returning one fully specified prompt per passed
        Doc instance.
    RETURNS (Callable[[Iterable[Doc]], Iterable[str]]): Callable producing prompt strings.
    """
    template = "What is {value} times three? Respond with the exact number."

    def prompt_template(docs: Iterable[Doc]) -> Iterable[str]:
        return [template.format(value=len(doc)) for doc in docs]

    return prompt_template


@registry.llm("spacy.DummyPrompt.v1")
def dummy_prompt() -> Callable[
    [minichain.backend.Backend, Iterable[str]], Iterable[str]
]:
    """Returns Callable prompting LLM API and returning responses.
    RETURNS (Callable[[minichain.backend.Backend, Iterable[str]], Iterable[str]]): Prompts LLM and returns responses.
    """

    def prompt(
        backend: minichain.backend.Backend, prompts: Iterable[str]
    ) -> Iterable[str]:
        @minichain.prompt(backend())
        def _prompt(model: minichain.backend, prompt_text: str) -> str:
            return model(prompt_text)

        return [_prompt(pr).run() for pr in prompts]

    return prompt


@registry.llm("spacy.DummyParse.v1")
def dummy_parse() -> Callable[
    [Iterable[Doc], Iterable[str], Optional[str]], Iterable[Doc]
]:
    """Returns Callable parsing LLM responses and updating Doc instances with the extracted information.
    RETURNS (Callable[[Iterable[Doc], Iterable[str], Optional[str]], Iterable[Doc]]): Callable parsing LLM responses and
        mapping them to Doc instances.
    """

    def prompt_parse(
        docs: Iterable[Doc],
        prompt_responses: Iterable[str],
        response_field: Optional[str],
    ) -> Iterable[Doc]:
        for doc, prompt_response in zip(docs, prompt_responses):
            if response_field:
                setattr(doc._, response_field, prompt_response)

        return docs

    return prompt_parse
