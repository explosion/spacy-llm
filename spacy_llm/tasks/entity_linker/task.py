from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type

import jinja2
from spacy import Language
from spacy.pipeline import EntityLinker
from spacy.tokens import Doc, Span
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
    ):
        """Default entity linking task.

        parse_responses (TaskResponseParser[Self]): Callable for parsing LLM responses for this task.
        prompt_example_type (Type[FewshotExample]): Type to use for fewshot examples.
        prompt_examples (Optional[List[FewshotExample]]): Optional list of few-shot examples to include in prompts.
        template (str): Prompt template passed to the model.
        scorer (Scorer): Scorer function.
        """
        super().__init__(
            parse_responses=parse_responses,
            prompt_example_type=prompt_example_type,
            template=template,
            prompt_examples=prompt_examples,
        )
        self._scorer = scorer
        self._candidate_selector: Optional[CandidateSelector] = None

        # Exclude mentions without candidates from prompt, if set.
        self._auto_nil = True
        # Store, per doc and entity, whether candidates could be found.
        self._has_ent_cands: List[List[bool]] = []

    def initialize(
        self,
        get_examples: Callable[[], Iterable["Example"]],
        nlp: Language,
        candidate_selector: CandidateSelector,
        n_prompt_examples: int = 0,
    ) -> None:
        """Initialize entity linking task.
        get_examples (Callable[[], Iterable["Example"]]): Callable that provides examples
            for initialization.
        nlp (Language): Language instance.
        candidate_selector (CandidateSelector): Factory for a candidate selection callable
            returning candidates for a given Span and context.
        n_prompt_examples (int): How many prompt examples to infer from the provided Example objects.
            0 by default. Takes all examples if set to -1.

        """
        super()._initialize(
            get_examples=get_examples,
            nlp=nlp,
            n_prompt_examples=n_prompt_examples,
            fetch_entity_info=self._fetch_entity_info,
        )
        self._candidate_selector = candidate_selector

    def generate_prompts(self, docs: Iterable[Doc], **kwargs) -> Iterable[str]:
        environment = jinja2.Environment()
        _template = environment.from_string(self._template)
        # Reset auto-nil attributes for new batch of docs.
        self._has_ent_cands = []

        for i_doc, doc in enumerate(docs):
            cands_ents, _ = self._fetch_entity_info(doc)
            # Determine which ents have candidates and should be included in prompt.
            has_cands = [
                {cand_ent.id for cand_ent in cand_ents} != {EntityLinker.NIL}
                or not self._auto_nil
                for cand_ents in cands_ents
            ]
            self._has_ent_cands.append(has_cands)

            # To improve: if a doc has no entities (with candidates), skip prompt altogether?
            yield _template.render(
                text=EntityLinkerTask.highlight_ents_in_text(doc, has_cands),
                mentions_str=", ".join(
                    [f"*{mention}*" for hc, mention in zip(has_cands, doc.ents) if hc]
                ),
                mentions=[ent.text for hc, ent in zip(has_cands, doc.ents) if hc],
                entity_descriptions=[
                    [ent.description for ent in ents]
                    for hc, ents in zip(has_cands, cands_ents)
                    if hc
                ],
                entity_ids=[
                    [ent.id for ent in ents]
                    for hc, ents in zip(has_cands, cands_ents)
                    if hc
                ],
                prompt_examples=self._prompt_examples,
            )

    def parse_responses(
        self, docs: Iterable[Doc], responses: Iterable[str]
    ) -> Iterable[Doc]:
        for i_doc, (doc, ent_spans) in enumerate(
            zip(docs, self._parse_responses(self, docs=docs, responses=responses))
        ):
            gen_nil_span: Callable[[Span], Span] = lambda ent: Span(  # noqa: E731
                doc=doc,
                start=ent.start,
                end=ent.end,
                label=ent.label,
                vector=ent.vector,
                vector_norm=ent.vector_norm,
                kb_id=EntityLinker.NIL,
            )

            # If numbers of ents parsed from LLM response + ents without candiates and number of ents in doc don't
            # align, skip doc (most likely LLM parsing failed, no guarantee KB IDs can be assigned to correct ents).
            # This can happen when the LLM fails to list solutions for all entities.
            all_entities_resolved = len(ent_spans) + sum(
                [not is_in_prompt for is_in_prompt in self._has_ent_cands[i_doc]]
            ) == len(doc.ents)

            # Fuse entities with (i. e. inferred by the LLM) and without candidates (i. e. auto-niled).
            # If entity was not included in prompt, as there were no candidates - fill in NIL for this entity.
            # If numbers of inferred and auto-niled entities don't line up with total number of entities, there is no
            # guaranteed way to assign a partially resolved list of entities
            # correctly.
            # Else: entity had candidates and was included in prompt - fill in resolved KB ID.
            ent_spans_iter = iter(ent_spans)
            doc.ents = [
                gen_nil_span(ent)
                if not (all_entities_resolved and self._has_ent_cands[i_doc][i_ent])
                else next(ent_spans_iter)
                for i_ent, ent in enumerate(doc.ents)
            ]

            yield doc

    def scorer(self, examples: Iterable[Example]) -> Dict[str, Any]:
        return self._scorer(examples)

    @property
    def _cfg_keys(self) -> List[str]:
        return ["_template"]

    @staticmethod
    def highlight_ents_in_text(
        doc: Doc, include_ents: Optional[List[bool]] = None
    ) -> str:
        """Highlights entities in doc text with **.
        doc (Doc): Doc whose entities are to be highlighted.
        include_ents (Optional[List[bool]]): Whether to include entities with the corresponding indices. If None, all
            are included.
        RETURNS (str): Text with highlighted entities.
        """
        if include_ents is not None and len(include_ents) != len(doc.ents):
            raise ValueError(
                f"`include_ents` has {len(include_ents)} entries, but {len(doc.ents)} are required."
            )

        text = doc.text
        i = 0
        for ent in doc.ents:
            # Skip if ent is not supposed to be included.
            if include_ents is not None and not include_ents[i]:
                continue

            text = (
                text[: ent.start_char + i * 2]
                + f"*{ent.text}*"
                + text[ent.end_char + i * 2 :]
            )
            i += 1

        return text

    def _require_candidate_selector(self) -> None:
        """Raises an error if candidate selector is not available."""
        if not self._candidate_selector:
            raise ValueError(
                "Candidate selector hasn't been initialized. Pass the corresponding config to "
                "[initialize.components.LLM_TASK_NAME.candidate_selector]."
            )

    def _fetch_entity_info(
        self, doc: Doc
    ) -> Tuple[List[List[Entity]], List[Optional[str]]]:
        """Fetches entity IDs & descriptions and determines solution numbers for entities in doc.
        doc (Doc): Doc to fetch entity descriptions and solution numbers for. If entities' KB IDs are not set,
            corresponding solution number will be None.
        Tuple[List[List[Entity]], List[Optional[str]]]: For each mention in doc: list of entity candidates,
            list of correct entity IDs.
        """
        self._require_candidate_selector()
        assert self._candidate_selector

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

    @property
    def has_ent_cands(self) -> List[List[bool]]:
        """Returns flags indicating whether documents' entities' have candidates in KB.
        RETURNS (List[List[bool]]): Flags indicating whether documents' entities' have candidates in KB.
        """
        return self._has_ent_cands
