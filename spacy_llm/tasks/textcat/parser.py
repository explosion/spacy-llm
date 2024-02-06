from typing import Dict, Iterable, List

from spacy.tokens import Doc
from wasabi import msg

from .task import TextCatTask


def parse_responses_v1_v2_v3(
    task: TextCatTask,
    shards: Iterable[Iterable[Doc]],
    responses: Iterable[Iterable[str]],
) -> Iterable[Iterable[Dict[str, float]]]:
    """Parses LLM responses for spacy.TextCat.v1, v2 and v3
    task (LemmaTask): Task instance.
    shards (Iterable[Iterable[Doc]]): Doc shards.
    responses (Iterable[Iterable[str]]): LLM responses.
    RETURNS (Iterable[Iterable[Dict[str, float]]]): TextCat scores per shard and class.
    """
    for response_for_doc in responses:
        results_for_doc: List[Dict[str, float]] = []

        for response in response_for_doc:
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

            results_for_doc.append(categories)

        yield results_for_doc
