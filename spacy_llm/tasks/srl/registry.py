from typing import Callable, Dict, List, Optional, Union, Any, Type

from .parser import parse_responses_v1
from .task import DEFAULT_SPAN_SRL_TEMPLATE_V1, SRLTask
from .util import SRLExample, score
from ...compat import Literal
from ...registry import registry
from ...ty import ExamplesConfigType, TaskResponseParser, Scorer, FewshotExample
from ...util import split_labels


@registry.llm_tasks("spacy.SRL.v1")
def make_srl_task(
    template: str = DEFAULT_SPAN_SRL_TEMPLATE_V1,
    parse_responses: Optional[TaskResponseParser[SRLTask]] = None,
    prompt_example_type: Optional[Type[FewshotExample]] = None,
    scorer: Optional[Scorer] = None,
    examples: ExamplesConfigType = None,
    labels: Union[List[str], str] = [],
    label_definitions: Optional[Dict[str, str]] = None,
    normalizer: Optional[Callable[[str], str]] = None,
    alignment_mode: Literal["strict", "contract", "expand"] = "contract",
    case_sensitive_matching: bool = True,
    single_match: bool = True,
    verbose: bool = False,
    predicate_key: str = "Predicate",
):
    """SRL.v1 task factory.

    template (str): Prompt template passed to the model.
    parse_responses (Optional[TaskResponseParser]): Callable for parsing LLM responses for this task.
    prompt_example_type (Optional[Type[FewshotExample]]): Type to use for fewshot examples.
    examples (Optional[Callable[[], Iterable[Any]]]): Optional callable that reads a file containing task examples for
        few-shot learning. If None is passed, then zero-shot learning will be used.
    scorer (Optional[Scorer]): Scorer function.
    labels (str): Comma-separated list of labels to pass to the template.
        Leave empty to populate it at initialization time (only if examples are provided).
    label_definitions (Optional[Dict[str, str]]): Map of label -> description
        of the label to help the language model output the entities wanted.
        It is usually easier to provide these definitions rather than
        full examples, although both can be provided.
    normalizer (Optional[Callable[[str], str]]): optional normalizer function.
    alignment_mode (Literal["strict", "contract", "expand"]): How character indices snap to token boundaries.
        Options: "strict" (no snapping), "contract" (span of all tokens completely within the character span),
        "expand" (span of all tokens at least partially covered by the character span).
        Defaults to "strict".
    case_sensitive_matching: Whether to search without case sensitivity.
    single_match (bool): If False, allow one substring to match multiple times in
        the text. If True, returns the first hit.
    verbose (bool): Verbose or not
    predicate_key (str): The str of Predicate in the template
    """
    labels_list = split_labels(labels)
    raw_examples = examples() if callable(examples) else examples
    example_type = prompt_example_type or SRLExample
    srl_examples = (
        [example_type(**eg) for eg in raw_examples] if raw_examples else None
    )

    return SRLTask(
        template=template,
        parse_responses=parse_responses or parse_responses_v1,
        prompt_example_type=example_type,
        prompt_examples=srl_examples,
        scorer=scorer or score,
        labels=labels_list,
        label_definitions=label_definitions,
        normalizer=normalizer,
        verbose=verbose,
        alignment_mode=alignment_mode,
        case_sensitive_matching=case_sensitive_matching,
        single_match=single_match,
        predicate_key=predicate_key,
    )
