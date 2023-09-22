import warnings
from typing import Callable, Dict, Iterable, List, Optional, Type

import jinja2
from spacy.language import Language
from spacy.tokens import Doc
from spacy.training import Example

from ...compat import Literal, Self
from ...ty import FewshotExample, Scorer, TaskResponseParser
from ..span import SpanTask
from ..templates import read_template
from .util import SRLExample

DEFAULT_SPAN_SRL_TEMPLATE_V1 = read_template("span-srl.v1")


class SRLTask(SpanTask):
    def __init__(
        self,
        template: str,
        parse_responses: TaskResponseParser[Self],
        prompt_example_type: Type[FewshotExample],
        prompt_examples: Optional[List[FewshotExample]],
        scorer: Scorer,
        labels: List[str],
        label_definitions: Optional[Dict[str, str]],
        normalizer: Optional[Callable[[str], str]],
        alignment_mode: Literal["strict", "contract", "expand"],  # noqa: F821
        case_sensitive_matching: bool,
        single_match: bool,
        verbose: bool,
        predicate_key: str,
    ):
        """
        template (str): Prompt template passed to the model.
        parse_responses (TaskResponseParser): Callable for parsing LLM responses for this task.
        prompt_example_type (Type[FewshotExample]): Type to use for fewshot examples.
        prompt_examples (Optional[List[FewshotExample]]): Optional list of few-shot examples to include in prompts.
        scorer (Scorer): Scorer function.
        labels (List[str]): List of labels to pass to the template.
            Leave empty to populate it at initialization time (only if examples are provided).
        label_definitions (Optional[Dict[str, str]]): Map of label -> description
            of the label to help the language model output the entities wanted.
            It is usually easier to provide these definitions rather than
            full examples, although both can be provided.
        spans_key (str): Key of the `Doc.spans` dict to save under.
        normalizer (Optional[Callable[[str], str]]): optional normalizer function.
        alignment_mode (str): "strict", "contract" or "expand".
        case_sensitive_matching (bool): Whether to search without case sensitivity.
        single_match (bool): If False, allow one substring to match multiple times in
            the text. If True, returns the first hit.
        description (str): A description of what to recognize or not recognize as entities.
        check_label_consistency (SpanTaskLabelCheck): Callable to check label consistency.
        """
        super().__init__(
            parse_responses=parse_responses,
            prompt_example_type=prompt_example_type,
            labels=labels,
            template=template,
            label_definitions=label_definitions,
            prompt_examples=prompt_examples,
            normalizer=normalizer,
            alignment_mode=alignment_mode,
            case_sensitive_matching=case_sensitive_matching,
            single_match=single_match,
            description=None,
            allow_overlap=False,
            check_label_consistency=SRLTask._check_label_consistency,
        )

        self._predicate_key = predicate_key
        self._verbose = verbose
        self._scorer = scorer
        self._check_extensions()

    @classmethod
    def _check_extensions(cls):
        """Add `predicates` extension if need be.
        Add `relations`  extension if need be."""
        if not Doc.has_extension("predicates"):
            Doc.set_extension("predicates", default=[])

        if not Doc.has_extension("relations"):
            Doc.set_extension("relations", default=[])

    def initialize(
        self,
        get_examples: Callable[[], Iterable["Example"]],
        nlp: Language,
        labels: List[str] = [],
        n_prompt_examples: int = 0,
    ) -> None:
        self._check_extensions()

        super()._initialize(
            get_examples=get_examples,
            nlp=nlp,
            labels=labels,
            n_prompt_examples=n_prompt_examples,
        )

    def generate_prompts(self, docs: Iterable[Doc], **kwargs) -> Iterable[str]:
        # todo Simplify after **kwargs ditching PR has been merged.
        environment = jinja2.Environment()
        _template = environment.from_string(self._template)
        for doc in docs:
            predicates = None
            if len(doc._.predicates):
                predicates = ", ".join([p["text"] for p in doc._.predicates])

            doc_examples = self._prompt_examples

            # check if there are doc-tailored examples
            if doc.has_extension("egs") and doc._.egs is not None and len(doc._.egs):
                doc_examples = doc._.egs

            prompt = _template.render(
                text=doc.text,
                labels=list(self._label_dict.values()),
                label_definitions=self._label_definitions,
                predicates=predicates,
                examples=doc_examples,
            )

            yield prompt

    @property
    def _cfg_keys(self) -> List[str]:
        return [
            "_label_dict",
            "_template",
            "_label_definitions",
            "_verbose",
            "_predicate_key",
            "_alignment_mode",
            "_case_sensitive_matching",
            "_single_match",
        ]

    def parse_responses(
        self, docs: Iterable[Doc], responses: Iterable[str]
    ) -> Iterable[Doc]:
        for doc, (predicates, relations) in zip(
            docs, self._parse_responses(self, docs, responses)
        ):
            doc._.predicates = predicates
            doc._.relations = relations
            yield doc

    def _extract_labels_from_example(self, example: Example) -> List[str]:
        if hasattr(example, "relations"):
            return [r.label for p, rs in example.relations for r in rs]
        return []

    @classmethod
    def _check_label_consistency(cls, task: Self) -> List[FewshotExample]:
        """Checks consistency of labels between examples and defined labels. Emits warning on inconsistency.

        Note: it's unusual for a SpanTask to have its own label consistency check implementation (and an example type
        not derived from SpanExample). This should be cleaned up and unified.

        task (SRLTask): SRLTask instance.
        RETURNS (List[FewshotExample]): List of SRLExamples with valid labels.
        """
        assert task.prompt_examples
        assert issubclass(task.prompt_example_type, SRLExample)

        srl_examples = [
            task.prompt_example_type(**eg.dict()) for eg in task.prompt_examples
        ]
        example_labels = {
            task.normalizer(r.label): r.label
            for example in srl_examples
            for p, rs in example.relations
            for r in rs
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

        # Return examples without non-declared roles. the roles within a predicate that have undeclared role labels
        # are discarded.
        return [
            example
            for example in [
                task.prompt_example_type(
                    text=example.text,
                    predicates=example.predicates,
                    relations=[
                        (
                            p,
                            [
                                r
                                for r in rs
                                if task.normalizer(r.label) in task.label_dict
                            ],
                        )
                        for p, rs in example.relations
                    ],
                )
                for example in srl_examples
            ]
        ]

    @property
    def predicate_key(self) -> str:
        return self._predicate_key

    @property
    def verbose(self) -> bool:
        return self._verbose
