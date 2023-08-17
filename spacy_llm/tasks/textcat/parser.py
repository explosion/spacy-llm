from typing import Dict, Iterable

from spacy.tokens import Doc
from wasabi import msg

from .task import TextCatTask


def parse_responses_v1_v2_v3(
    task: TextCatTask, docs: Iterable[Doc], responses: Iterable[str]
) -> Iterable[Dict[str, float]]:
    """Parses LLM responses for spacy.TextCat.v1, v2 and v3
    task (LemmaTask): Task instance.
    docs (Iterable[Doc]): Corresponding Doc instances.
    responses (Iterable[str]): LLM responses.
    RETURNS (Dict[str, float]): TextCat scores per class.
    """
    for response in responses:
        categories: Dict[str, float]
        response = response.strip()
        if task.use_binary:
            # Binary classification: We only have one label
            label: str = list(task.label_dict.values())[0]
            score = 1.0 if response.upper() == "POS" else 0.0
            categories = {label: score}
        else:
            # Multilabel classification
            categories = {label: 0.0 for label in task.label_dict.values()}

            pred_labels = response.split(",")
            if task.exclusive_classes and len(pred_labels) > 1:
                # Don't use anything but raise a debug message
                # Don't raise an error. Let user abort if they want to.
                msg.text(
                    f"LLM returned multiple labels for this exclusive task: {pred_labels}.",
                    " Will store an empty label instead.",
                    show=task.verbose,
                )
                pred_labels = []

            for pred in pred_labels:
                if task.normalizer(pred.strip()) in task.label_dict:
                    category = task.label_dict[task.normalizer(pred.strip())]
                    categories[category] = 1.0

        yield categories
