from typing import Iterable, List, Union


def split_labels(labels: Union[str, Iterable[str]]) -> List[str]:
    """Split a comma-separated list of labels.
    If input is a list already, just strip each entry of the list"""
    labels = labels.split(",") if isinstance(labels, str) else labels
    return [label.strip() for label in labels]
