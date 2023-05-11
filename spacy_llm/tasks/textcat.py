from typing import Any, Callable, Dict, Iterable, Optional

import jinja2
from pydantic import BaseModel
from spacy.tokens import Doc
from wasabi import msg

from ..registry import lowercase_normalizer, registry


class TextCatExample(BaseModel):
    text: str
    answer: str


@registry.llm_tasks("spacy.TextCat.v1")
class TextCatTask:
    _TEMPLATE_STR = """
{%- if labels|length == 1 -%}
{%- set label = labels[0] -%}
Classify whether the text below belongs to the {{ label }} category or not.
If it is a {{ label }}, answer `POS`. If it is not a {{ label }}, answer `NEG`.
{%- else -%}
Classify the text below to any of the following labels: {{ labels|join(", ") }}
{# whitespace #}
{%- if exclusive_classes -%}
The task is exclusive, so only choose one label from what I provided.
{%- else -%}
The task is non-exclusive, so you can provide more than one label as long as
they're comma-delimited. For example: Label1, Label2, Label3.
{%- if allow_none -%}
{# whitespace #}
If the text cannot be classified into any of the provided labels, answer `==NONE==`.
{%- endif -%}
{%- endif -%}
{# whitespace #}
{%- endif -%}
{# whitespace #}
{%- if examples -%}
{# whitespace #}
Below are some examples (only use these as a guide):
{# whitespace #}
{# whitespace #}
{%- for example in examples -%}
{# whitespace #}
Text:
'''
{{ example.text }}
'''
{# whitespace #}
{{ example.answer }}
{# whitespace #}
{%- endfor -%}
{%- endif -%}
{# whitespace #}
{# whitespace #}
Here is the text that needs classification
{# whitespace #}
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
        allow_none: bool = True,
        verbose: bool = False,
    ):
        """Default TextCat task.

        You can use either binary or multilabel text classification based on the
        labels you provide.

        If a single label is provided, binary classification
        will be used. The label will get a score of `0` or `1` in `doc.cats`.

        If a comma-separated list of labels is provided, multilabel
        classification will be used. The document labels in `doc.cats` will be a
        dictionary of strings and their score.

        Lastly, you can toggle between exclusive or no-exclusive text
        categorization by passing a flag to the `exclusive_classes` parameter.

        labels (str): Comma-separated list of labels to pass to the template. This task
            assumes binary classification if a single label is provided.
        examples (Optional[Callable[[], Iterable[Any]]]): Optional callable that
            reads a file containing task examples for few-shot learning. If None is
            passed, then zero-shot learning will be used.
        normalizer (Optional[Callable[[str], str]]): Optional normalizer function.
        exclusive_classes (bool): If True, require the language model to suggest only one
            label per class. This is automatically set when using binary classification.
        allow_none (bool): if True, there might be cases where no label is applicable.
        verbose (bool): If True, show extra information.
        """
        self._normalizer = normalizer if normalizer else lowercase_normalizer()
        self._label_dict = {
            self._normalizer(label): label for label in labels.split(",")
        }
        self._examples = (
            [TextCatExample(**eg) for eg in examples()] if examples else None
        )
        # Textcat configuration
        self._use_binary = True if len(self._label_dict) == 1 else False
        self._exclusive_classes = exclusive_classes
        self._allow_none = allow_none
        self._verbose = verbose

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
                text=doc.text,
                labels=list(self._label_dict.values()),
                examples=self._examples,
                exclusive_classes=self._exclusive_classes,
                allow_none=self._allow_none,
            )
            yield prompt

    def _format_response(self, response: str) -> Dict[str, float]:
        """Parse raw string response into a structured format

        The returned dictionary contains the labels mapped to their score.
        """
        categories: Dict[str, float]
        if self._use_binary:
            # Binary classification: We only have one label
            label: str = list(self._label_dict.values())[0]
            score = 1.0 if response.upper() == "POS" else 0.0
            categories = {label: score}
        else:
            # Multilabel classification
            categories = {label: 0.0 for label in self._label_dict.values()}

            pred_labels = response.split(",")
            if self._exclusive_classes and len(pred_labels) > 1:
                # Don't use anything but raise a debug message
                # Don't raise an error. Let user abort if they want to.
                msg.text(
                    f"LLM returned multiple labels for this exclusive task: {pred_labels}.",
                    " Will store an empty label instead.",
                    show=self._verbose,
                )
                pred_labels = []

            for pred in pred_labels:
                if self._normalizer(pred.strip()) in self._label_dict:
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
