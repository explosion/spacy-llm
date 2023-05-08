from typing import Callable, Dict, Iterable, Optional, Tuple

import jinja2
import spacy
from spacy.tokens import Doc
from wasabi import msg

from ..registry import noop_normalizer


@spacy.registry.llm_tasks("spacy.TextCatZeroShot.v1")
def textcat_zeroshot_task(
    labels: str,
    normalizer: Optional[Callable[[str], str]] = None,
    exclusive_classes: bool = False,
) -> Tuple[
    Callable[[Iterable[Doc]], Iterable[str]],
    Callable[[Iterable[Doc], Iterable[str]], Iterable[Doc]],
]:
    """Default TextCat template for zero-shot annotation

    You can use either binary or multilabel text classification based on the
    labels you provide. If a single label is provided, binary classification is
    done. The output documents will have POS and NEG labels for positive and
    negative classes respectively.

    Provide a comma-separated list of labels if you want to use multilabel
    classification. You can toggle between exclusive or non-exclusive text
    categorization by passing a flag to the `exclusive_classes` parameter.

    labels (str): comma-separated list of labels to pass to the template. This task
        assumes binary classification if a single label is provided.
    normalizer (Optional[Callable[[str], str]]): optional normalizer function.
    exclusive_classes (bool): if True, require the language model to suggest only one
        label per class. This is automatically set when using binary classification.

    RETURNS (Tuple[Callable[[Iterable[Doc]], Iterable[str]], Any]): templating Callable, parsing Callable.
    """

    # Set up the labels and the parameters of the task
    if not normalizer:
        normalizer = noop_normalizer()

    binary_textcat = True if len(labels.split(",")) == 1 else False
    label_dict = {normalizer(label): label for label in labels.split(",")}
    if binary_textcat and not exclusive_classes:
        # This doesn't really affect the template since exclusivity only
        # matters for multilabel. But it's good to call out this via a warning.
        msg.warn(
            "Binary classification should always be exclusive. Setting "
            "`exclusive_classes` parameter to True"
        )
        exclusive_classes = True

    # Set up the template and its sub-components
    _tpl_binary = """
    Classify whether the text below belongs to the {{ label }} category or not.
    If it is a {{ label }}, answer `POS`. If it is not a {{ label }}, answer
    `NEG`.
    """

    _tpl_multilabel = """
    Classify the text below to any of the following labels: {{ labels|join(", ") }}
    """

    _tpl_exclusive = """
    The task is exclusive, so only choose one label from what I provided
    """

    _tpl_nonexclusive = """
    The task is non-exclusive, so you can provide more than one label as long as
    they're comma-delimited. For example: Label1, Label2, Label3
    """

    template = """
    {% if labels|length == 1 %}    
    {% set label = labels[0] %}
    {binary}
    {% else %}
    {multilabel}
    {% if exclusive_classes %}
    {exclusive}
    {% else %}
    {nonexclusive}
    {% endif %}
    {% endif %}
    {# whitespace #}
    Text:
    '''
    {{ text }}
    '''
    """.format(
        binary=_tpl_binary,
        multilabel=_tpl_multilabel,
        exlusive=_tpl_exclusive,
        nonexclusive=_tpl_nonexclusive,
    )

    def prompt_template(docs: Iterable[Doc]) -> Iterable[str]:
        environment = jinja2.Environment()
        _template = environment.from_string(template)
        for doc in docs:
            prompt = _template.render(text=doc.text, labels=list(label_dict.values()))
            yield prompt

    def _format_response(response: str) -> Dict[str, float]:
        """Parse raw string response into a structured format

        If using binary classification, the categories will be `POS` and `NEG`
        for the positive and negative labels respectively.
        """
        labels = {"POS", "NEG"} if binary_textcat else set(label_dict.values())
        categories = {label: 0 for label in labels}
        for label in response.split(","):
            cat = label.upper() if binary_textcat else label_dict(normalizer(label))
            categories[cat] = 1
        return categories

    def prompt_parse(
        docs: Iterable[Doc], prompt_responses: Iterable[str]
    ) -> Iterable[Doc]:
        for doc, prompt_response in zip(docs, prompt_response):
            cats = _format_response(prompt_response)
            doc.cats = cats
            yield doc

    return prompt_template, prompt_parse
