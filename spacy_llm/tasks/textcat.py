from typing import Any, Callable, Dict, Iterable, Optional

import jinja2
from spacy.tokens import Doc
from wasabi import msg

from ..registry import noop_normalizer, registry


@registry.llm_tasks("spacy.TextCat.v1")
class TextCatTask:
    _TEMPLATE_STR = """
{% if labels|length == 1 %}
{% set label = labels[0] %}
Classify whether the text below belongs to the {{ label }} category or not.
If it is a {{ label }}, answer `POS`. If it is not a {{ label }}, answer
`NEG`.
{% else %}
Classify the text below to any of the following labels: {{ labels|join(", ") }}
{% if exclusive_classes %}
The task is exclusive, so only choose one label from what I provided
{% else %}
The task is non-exclusive, so you can provide more than one label as long as
they're comma-delimited. For example: Label1, Label2, Label3
{% endif %}
{% endif %}
{# whitespace #}
{% if examples %}
Below are some examples (only use these as a guide):
{# whitespace #}
{# whitespace #}
{% for example in examples %}
Text:
'''
{{ example['text'] }}
'''
{# whitespace #}
{{ example['answer']}}
{# whitespace #}
{% endfor %}
{% endif %}
{# whitespace #}
Here is the text that needs classification
{# whitespace #}
Text:
'''
{{ text }}
'''
    """

    def __init__(
        self,
        labels: str,
        examples: Optional[Callable[[], Iterable[Any]]] = None,
        normalizer: Optional[Callable[[str], str]] = None,
        exclusive_classes: bool = False,
    ):
        """Default TextCat task for LLM annotation

        You can use either binary or multilabel text classification based on the
        labels you provide.

        If a single label is provided, binary classification
        will be used. Positive examples will contain the original label, and
        negative examples will contain no labels.

        If a comma-separated list of labels is provided, multilabel
        classification will be used. The document labels will be a dictionary of
        strings and their score.

        Lastly, you can toggle between exclusive or no-exclusive text
        categorization by passing a flag to the `exclusive_classes` parameter.

        labels (str): comma-separated list of labels to pass to the template. This task
            assumes binary classification if a single label is provided.
        examples (Optional[Callable[[], Iterable[Any]]]): optional callable that
            reads a file containing task examples for few-shot learning. If None is
            passed, then zero-shot learning will be used.
        normalizer (Optional[Callable[[str], str]]): optional normalizer function.
        exclusive_classes (bool): if True, require the language model to suggest only one
            label per class. This is automatically set when using binary classification.
        """
        self._normalizer = normalizer if normalizer else noop_normalizer()
        self._label_dict = {
            self._normalizer(label): label for label in labels.split(",")
        }
        self._examples = examples() if examples else None
        # Textcat configuration
        self._use_binary = True if len(labels.split(",")) == 1 else False
        self._exclusive_classes = exclusive_classes

        if self._use_binary and not self._exclusive_classes:
            msg.warn(
                "Binary classification should always be exclusive. Setting "
                "the `exclusive_classes` parameter to True."
            )
            self._exclusive_classes = True

    def generate_prompts(self, docs: Iterable[Doc]) -> Iterable[str]:
        environment = jinja2.Environment()
        _template = environment.from_string(self._TEMPLATE_STR)
        for doc in docs:
            prompt = _template.render(
                text=doc.text, labels=list(self._label_dict.values())
            )
            yield prompt

    def _format_response(self, response: str) -> Dict[str, float]:
        """Parse raw string response into a structured format

        If using binary classification, positive examples will contain the
        original label, and negative examples will contain no labels.

        If using multilabel classification, the document labels will be a
        dictionary of strings and their score.
        """
        if self._use_binary:
            # Binary classification: We only have one label
            label: str = list(self._label_dict.values())[0]
            categories = {label: 1.0} if response.upper() == "POS" else {}
        else:
            # Multilabel classification
            categories = {label: 0 for label in self._label_dict.values()}
            for pred in response.split(","):
                category = self._label_dict[self._normalizer(pred.strip())]
                categories[category] = 1.0
        return categories

    def parse_responses(
        self, docs: Iterable[Doc], responses: Iterable[str]
    ) -> Iterable[Doc]:
        for doc, prompt_response in zip(docs, responses):
            cats = self._format_response(prompt_response)
            doc.cats = cats
            yield doc
