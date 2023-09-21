import warnings
from typing import List

from .examples import SpanCoTExample, SpanExample
from .task import SpanTask


def check_label_consistency(task: SpanTask) -> List[SpanExample]:
    """Checks consistency of labels between examples and defined labels for non-CoT SpanTask. Emits warning on
    inconsistency.
    RETURNS (List[SpanExample]): List of SpanExamples with valid labels.
    """
    assert task.prompt_examples
    assert issubclass(task.prompt_example_type, SpanExample)

    example_labels = {
        task.normalizer(key): key
        for example in task.prompt_examples
        for key in example.entities
    }
    unspecified_labels = {
        example_labels[key]
        for key in (set(example_labels.keys()) - set(task.label_dict.keys()))
    }
    if not set(example_labels.keys()) <= set(task.label_dict.keys()):
        warnings.warn(
            f"Examples contain labels that are not specified in the task configuration. The latter contains the "
            f"following labels: {sorted(list(set(task.label_dict.values())))}. Labels in examples missing from "
            f"the task configuration: {sorted(list(unspecified_labels))}. Please ensure your label specification "
            f"and example labels are consistent."
        )

    # Return examples without non-declared labels. If an example only has undeclared labels, it is discarded.
    return [
        example
        for example in [
            task.prompt_example_type(
                text=example.text,
                entities={
                    label: entities
                    for label, entities in example.entities.items()
                    if task.normalizer(label) in task.label_dict
                },
            )
            for example in task.prompt_examples
        ]
        if len(example.entities)
    ]


def check_label_consistency_cot(task: SpanTask) -> List[SpanCoTExample]:
    """Checks consistency of labels between examples and defined labels for CoT version of SpanTask. Emits warning on
    inconsistency.
    RETURNS (List[SpanExampleCoT]): List of SpanExamples with valid labels.
    """
    assert task.prompt_examples
    assert issubclass(task.prompt_example_type, SpanCoTExample)

    null_labels = {
        task.normalizer(entity.label): entity.label
        for example in task.prompt_examples
        for entity in example.spans
        if not entity.is_entity
    }

    if len(null_labels) > 1:
        warnings.warn(
            f"Negative examples contain multiple negative labels: {', '.join(null_labels.keys())}."
        )

    example_labels = {
        task.normalizer(entity.label): entity.label
        for example in task.prompt_examples
        for entity in example.spans
        if entity.is_entity
    }

    unspecified_labels = {
        example_labels[key]
        for key in (set(example_labels.keys()) - set(task.label_dict.keys()))
    }
    if not set(example_labels.keys()) <= set(task.label_dict.keys()):
        warnings.warn(
            f"Examples contain labels that are not specified in the task configuration. The latter contains the "
            f"following labels: {sorted(list(set(task.label_dict.values())))}. Labels in examples missing from "
            f"the task configuration: {sorted(list(unspecified_labels))}. Please ensure your label specification "
            f"and example labels are consistent."
        )

    # Return examples without non-declared labels. If an example only has undeclared labels, it is discarded.
    include_labels = dict(task.label_dict)
    include_labels.update(null_labels)

    return [
        example
        for example in [
            task.prompt_example_type(
                text=example.text,
                spans=[
                    entity
                    for entity in example.spans
                    if task.normalizer(entity.label) in include_labels
                ],
            )
            for example in task.prompt_examples
        ]
        if len(example.spans)
    ]
