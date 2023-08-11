from typing import Dict, Iterable

from wasabi import msg


def parse_responses_v1_v2_v3(
    responses: Iterable[str], **kwargs
) -> Iterable[Dict[str, float]]:
    """Parses LLM responses for spacy.TextCat.v1.
    responses (Iterable[str]): LLM responses.
    kwargs (Dict[str, Any]): Additional mandatory arguments:
        use_binary (bool): Whether to use  binary classification.
        label_dict (Dict[str, str]): Mapping of normalized to non-normalized labels.
        normalizer (Callable[[str], str]): normalizer function.
        exclusive_classes (bool): If True, require the language model to suggest only one label per class. This is
            automatically set when using binary classification.
        verbose (bool): Controls the verbosity of the task.
    RETURNS (Dict[str, float]): TextCat scores per class.
    """
    for response in responses:
        categories: Dict[str, float]
        response = response.strip()
        if kwargs["use_binary"]:
            # Binary classification: We only have one label
            label: str = list(kwargs["label_dict"].values())[0]
            score = 1.0 if response.upper() == "POS" else 0.0
            categories = {label: score}
        else:
            # Multilabel classification
            categories = {label: 0.0 for label in kwargs["label_dict"].values()}

            pred_labels = response.split(",")
            if kwargs["exclusive_classes"] and len(pred_labels) > 1:
                # Don't use anything but raise a debug message
                # Don't raise an error. Let user abort if they want to.
                msg.text(
                    f"LLM returned multiple labels for this exclusive task: {pred_labels}.",
                    " Will store an empty label instead.",
                    show=kwargs["verbose"],
                )
                pred_labels = []

            for pred in pred_labels:
                if kwargs["normalizer"](pred.strip()) in kwargs["label_dict"]:
                    category = kwargs["label_dict"][kwargs["normalizer"](pred.strip())]
                    categories[category] = 1.0

        yield categories
