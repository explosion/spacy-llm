from spacy.errors import ErrorsWithCodes


class Warnings(metaclass=ErrorsWithCodes):
    W001 = (
        "The '{legacy_task}' task is deprecated and will be removed in a future version. "
        "Use the '{new_task}' task instead."
    )
