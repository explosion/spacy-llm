from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type

import jinja2
from spacy import Language
from spacy.tokens import Doc
from spacy.training import Example

from ...compat import Self
from ...ty import FewshotExample, Scorer, TaskResponseParser
from ..builtin_task import BuiltinTask
from ..templates import read_template
from .ty import CandidateSelector, Entity

DEFAULT_EL_TEMPLATE_V1 = read_template("entity_linker.v1")


class EntityLinkerTask(BuiltinTask):
    def __init__(
        self,
        parse_responses: TaskResponseParser[Self],
        prompt_example_type: Type[FewshotExample],
        prompt_examples: Optional[List[FewshotExample]],
        template: str,
        scorer: Scorer,
        candidate_selector: CandidateSelector,
    ):
        """Default entity linking task.

        parse_responses (TaskResponseParser[Self]): Callable for parsing LLM responses for this task.
        prompt_example_type (Type[FewshotExample]): Type to use for fewshot examples.
        prompt_examples (Optional[List[FewshotExample]]): Optional list of few-shot examples to include in prompts.
        template (str): Prompt template passed to the model.
        scorer (Scorer): Scorer function.
        candidate_selector (CandidateSelector): Factory for a candidate selection callable
            returning candidates for a given Span and context.
        """
        super().__init__(
            parse_responses=parse_responses,
            prompt_example_type=prompt_example_type,
            template=template,
            prompt_examples=prompt_examples,
        )
        self._scorer = scorer
        self._candidate_selector = candidate_selector

    def initialize(
        self,
        get_examples: Callable[[], Iterable["Example"]],
        nlp: Language,
        n_prompt_examples: int = 0,
    ) -> None:
        super()._initialize(
            get_examples=get_examples,
            nlp=nlp,
            n_prompt_examples=n_prompt_examples,
            fetch_entity_info=self._fetch_entity_info,
        )

    def generate_prompts(self, docs: Iterable[Doc], **kwargs) -> Iterable[str]:
        environment = jinja2.Environment()
        _template = environment.from_string(self._template)
        for doc in docs:
            cands_ents, _ = self._fetch_entity_info(doc)
            yield _template.render(
                text=EntityLinkerTask.highlight_ents_in_text(doc).text,
                mentions_str=", ".join([f"*{mention}*" for mention in doc.ents]),
                mentions=[ent.text for ent in doc.ents],
                entity_descriptions=[
                    [cand_ent.description for cand_ent in cands_ent]
                    for cands_ent in cands_ents
                ],
                entity_ids=[
                    [cand_ent.id for cand_ent in cands_ent] for cands_ent in cands_ents
                ],
                prompt_examples=self._prompt_examples,
            )

    def parse_responses(
        self, docs: Iterable[Doc], responses: Iterable[str]
    ) -> Iterable[Doc]:
        for doc, ent_spans in zip(
            docs, self._parse_responses(self, docs=docs, responses=responses)
        ):
            doc.ents = ent_spans
            yield doc

    def scorer(self, examples: Iterable[Example]) -> Dict[str, Any]:
        return self._scorer(examples)

    @property
    def _cfg_keys(self) -> List[str]:
        return ["_template"]

    @staticmethod
    def highlight_ents_in_text(doc: Doc) -> Doc:
        """Highlights entities in doc text with **.
        doc (Doc): Doc whose entities are to be highlighted.
        RETURNS (str): Doc text with highlighted entities.
        """
        text = doc.text
        for i, ent in enumerate(doc.ents):
            text = (
                text[: ent.start_char + i * 2]
                + f"*{ent.text}*"
                + text[ent.end_char + i * 2 :]
            )

        # Unclean tokenization doesn't matter here, as only the raw text is used downstream currently.
        return Doc(doc.vocab, words=text.split())

    def _fetch_entity_info(
        self, doc: Doc
    ) -> Tuple[List[List[Entity]], List[Optional[str]]]:
        """Fetches entity IDs & descriptions and determines solution numbers for entities in doc.
        doc (Doc): Doc to fetch entity descriptions and solution numbers for. If entities' KB IDs are not set,
            corresponding solution number will be None.
        Tuple[List[List[EntityCandidate]], List[Optional[str]]]: For each mention in doc: list of entity candidates,
            list of correct entity IDs.
        """
        cands_per_ent: Iterable[Iterable[Entity]] = self._candidate_selector(doc.ents)
        cand_entity_info: List[List[Entity]] = []
        correct_ent_ids: List[Optional[str]] = []

        for ent, cands in zip(doc.ents, cands_per_ent):
            cands_for_ent: List[Entity] = list(cands)

            # No KB ID known: In this case there is no guarantee that the correct entity description will be included.
            if ent.kb_id == 0:
                correct_ent_ids.append(None)
            # Correct entity not in suggested candidates: fetch description explicitly.
            elif ent.kb_id not in {cand.id for cand in cands_for_ent}:
                cands_for_ent.append(
                    Entity(
                        id=ent.kb_id_,
                        description=self._candidate_selector.get_entity_description(
                            ent.kb_id_
                        ),
                    )
                )
            correct_ent_ids.append(ent.kb_id_)

            cand_entity_info.append(cands_for_ent)

        return cand_entity_info, correct_ent_ids
