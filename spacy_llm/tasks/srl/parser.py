import re
from typing import Iterable, List, Tuple, Any, Dict

from pydantic import ValidationError
from spacy.tokens import Doc
from wasabi import msg

from ..util.parsing import find_substrings
from .task import SRLTask
from .util import PredicateItem, RoleItem, SpanItem


def _format_response(task: SRLTask, arg_lines) -> List[Tuple[str, str]]:
    """Parse raw string response into a structured format.
    task (SRLTask): Task to format responses for.
    arg_lines ():
    RETURNS (List[Tuple[str, str]]): Formatted response.
    """
    output = []
    # this ensures unique arguments in the sentence for a predicate
    found_labels = set()
    for line in arg_lines:
        try:
            if line.strip() and ":" in line:
                label, phrase = line.strip().split(":", 1)

                # label is of the form "ARG-n (def)"
                label = label.split("(")[0].strip()

                # strip any surrounding quotes
                phrase = phrase.strip("'\" -")

                norm_label = task.normalizer(label)
                if norm_label in task.label_dict and norm_label not in found_labels:
                    if phrase.strip():
                        _phrase = phrase.strip()
                        found_labels.add(norm_label)
                        output.append((task.label_dict[norm_label], _phrase))
        except ValidationError:
            msg.warn(
                "Validation issue",
                line,
                show=task.verbose,
            )
    return output


def parse_responses_v1(
    task: SRLTask, docs: Iterable[Doc], responses: Iterable[str]
) -> Iterable[Tuple[List[Dict[str, Any]], List[Tuple[Dict[str, Any], List[Dict[str, Any]]]]]]:
    """
    Parse LLM response by extracting predicate-arguments blocks from the generate response.
    For example,
    LLM response for doc: "A sentence with multiple predicates (p1, p2)"

    Step 1: Extract the Predicates for the Text
    Predicates: p1, p2

    Step 2: For each Predicate, extract the Semantic Roles in 'Text'
    Text: A sentence with multiple predicates (p1, p2)
    Predicate: p1
    ARG-0: a0_1
    ARG-1: a1_1
    ARG-M-TMP: a_t_1
    ARG-M-LOC: a_l_1

    Predicate: p2
    ARG-0: a0_2
    ARG-1: a1_2
    ARG-M-TMP: a_t_2

    So the steps in the parsing are to first find the text boundaries for the information
    of each predicate. This is done by identifying the lines "Predicate: p1" and "Predicate: p2",
    which gives us the text for each predicate as follows:

    Predicate: p1
    ARG-0: a0_1
    ARG-1: a1_1
    ARG-M-TMP: a_t_1
    ARG-M-LOC: a_l_1

        and,

    Predicate: p2
    ARG-0: a0_2
    ARG-1: a1_2
    ARG-M-TMP: a_t_2

    Once we separate these out, then it is a matter of parsing line by line to extract the predicate
    and its args for each predicate block

    """
    for doc, prompt_response in zip(docs, responses):
        predicates: List[Dict[str, Any]] = []
        relations: List[Tuple[Dict[str, Any], List[Dict[str, Any]]]] = []
        lines = prompt_response.split("\n")

        # match lines that start with {Predicate:, Predicate 1:, Predicate1:}
        pred_patt = r"^" + re.escape(task.predicate_key) + r"\b\s*\d*[:\-\s]"
        pred_indices, pred_lines = zip(
            *[(i, line) for i, line in enumerate(lines) if re.search(pred_patt, line)]
        )

        pred_indices = list(pred_indices)

        # extract the predicate strings
        pred_strings = [line.split(":", 1)[1].strip("'\" ") for line in pred_lines]

        # extract the line ranges (s, e) of predicate's content.
        # then extract the pred content lines using the ranges
        pred_indices.append(len(lines))
        pred_ranges = zip(pred_indices[:-1], pred_indices[1:])
        pred_contents = [lines[s:e] for s, e in pred_ranges]

        # assign the spans of the predicates and args
        # then create ArgRELItem from the identified predicates and arguments
        for pred_str, pred_content_lines in zip(pred_strings, pred_contents):
            pred_offsets = list(
                find_substrings(
                    doc.text, [pred_str], case_sensitive=True, single_match=True
                )
            )

            # ignore the args if the predicate is not found
            if len(pred_offsets):
                p_start_char, p_end_char = pred_offsets[0]
                pred_item = PredicateItem(
                    text=pred_str, start_char=p_start_char, end_char=p_end_char
                ).dict()
                predicates.append(pred_item)

                roles = []

                for label, phrase in _format_response(task, pred_content_lines):
                    arg_offsets = find_substrings(
                        doc.text,
                        [phrase],
                        case_sensitive=task.case_sensitive_matching,
                        single_match=task.single_match,
                    )
                    for start, end in arg_offsets:
                        arg_item = SpanItem(
                            text=phrase, start_char=start, end_char=end
                        ).dict()
                        arg_rel_item = RoleItem(
                            predicate=pred_item, role=arg_item, label=label
                        ).dict()
                        roles.append(arg_rel_item)

                relations.append((pred_item, roles))

        yield predicates, relations
