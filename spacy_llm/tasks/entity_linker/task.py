from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type

from spacy import Language, Vocab
from spacy.pipeline import EntityLinker
from spacy.tokens import Doc, Span
from spacy.training import Example

from ...compat import Self
from ...ty import FewshotExample, Scorer, ShardMapper, ShardReducer, TaskResponseParser
from ..builtin_task import BuiltinTask
from ..templates import read_template
from .ty import CandidateSelector, Entity, InitializableCandidateSelector

DEFAULT_EL_TEMPLATE_V1 = read_template("entity_linker.v1")


class EntityLinkerTask(BuiltinTask):
    def __init__(
        self,
        parse_responses: TaskResponseParser[Self],
        prompt_example_type: Type[FewshotExample[Self]],
        prompt_examples: Optional[List[FewshotExample[Self]]],
        template: str,
        shard_mapper: ShardMapper,
        shard_reducer: ShardReducer[Self],
        scorer: Scorer,
    ):
        """Default entity linking task.

        parse_responses (TaskResponseParser[Self]): Callable for parsing LLM responses for this task.
        prompt_example_type (Type[FewshotExample[Self]]): Type to use for fewshot examples.
        prompt_examples (Optional[List[FewshotExample[Self]]]): Optional list of few-shot examples to include in prompts.
        template (str): Prompt template passed to the model.
        shard_mapper (ShardMapper): Maps docs to shards if they don't fit into the model context.
        shard_reducer (ShardReducer[Self]): Reduces doc shards back into one doc instance.
        scorer (Scorer): Scorer function.
        """
        super().__init__(
            parse_responses=parse_responses,
            prompt_example_type=prompt_example_type,
            template=template,
            prompt_examples=prompt_examples,
            shard_mapper=shard_mapper,
            shard_reducer=shard_reducer,
        )
        self._scorer = scorer
        self._candidate_selector: Optional[CandidateSelector] = None

        # Exclude mentions without candidates from prompt, if set. Mostly used for internal debugging.
        self._auto_nil = True
        # Store, per doc and entity, whether candidates could be found and candidates themselves.
        self._has_ent_cands_by_doc: List[List[bool]] = []
        self._ents_cands_by_doc: List[List[List[Entity]]] = []
        self._has_ent_cands_by_shard: List[List[List[bool]]] = []
        self._ents_cands_by_shard: List[List[List[List[Entity]]]] = []
        self._n_shards: Optional[int] = None

    def initialize(
        self,
        get_examples: Callable[[], Iterable["Example"]],
        nlp: Language,
        candidate_selector: Optional[CandidateSelector] = None,
        n_prompt_examples: int = 0,
    ) -> None:
        """Initialize entity linking task.
        get_examples (Callable[[], Iterable["Example"]]): Callable that provides examples
            for initialization.
        nlp (Language): Language instance.
        candidate_selector (Optional[CandidateSelector]): Factory for a candidate selection callable returning
            candidates for a given Span and context. If candidate selector hasn't been set explicitly before with
            .set_candidate_selector(), it has to be provided here - otherwise an error will be raised.
        n_prompt_examples (int): How many prompt examples to infer from the provided Example objects.
            0 by default. Takes all examples if set to -1.

        """
        super()._initialize(
            get_examples=get_examples,
            nlp=nlp,
            n_prompt_examples=n_prompt_examples,
            fetch_entity_info=self.fetch_entity_info,
        )
        if candidate_selector:
            self.set_candidate_selector(candidate_selector, nlp.vocab)
        elif self._candidate_selector is None:
            raise ValueError(
                "candidate_selector has to be provided when initializing the LLM component with the "
                "entity_linking task."
            )

    def set_candidate_selector(
        self, candidate_selector: CandidateSelector, vocab: Vocab
    ) -> None:
        """Sets candidate selector instance."""
        self._candidate_selector = candidate_selector
        if isinstance(self._candidate_selector, InitializableCandidateSelector):
            self._candidate_selector.initialize(vocab)

    def _preprocess_docs_for_prompt(self, docs: Iterable[Doc]) -> Iterable[Doc]:
        (
            self._ents_cands_by_doc,
            self._has_ent_cands_by_doc,
        ) = self._find_entity_candidates(docs)
        # Reset shard-wise candidate info. Will be set for each shard individually in _get_prompt_data(). We cannot
        # update it here, as we don't know yet how the shards will look like.
        self._ents_cands_by_shard = [[] * len(self._ents_cands_by_doc)]
        self._has_ent_cands_by_shard = [[] * len(self._ents_cands_by_doc)]
        self._n_shards = None
        return [
            EntityLinkerTask.highlight_ents_in_doc(doc, self._has_ent_cands_by_doc[i])
            for i, doc in enumerate(docs)
        ]

    def _find_entity_candidates(
        self, docs: Iterable[Doc]
    ) -> Tuple[List[List[List[Entity]]], List[List[bool]]]:
        """Determine entity candidates for all entity mentions in docs.
        docs (Iterable[Doc]): Docs with entities to select candidates for.
        RETURNS (Tuple[List[List[List[Entity]]], List[List[bool]]]): (1) list of candidate entities for each doc and
            entity, (2) list of flag whether candidates could be found per each doc and entitiy.
        """
        ents_cands: List[List[List[Entity]]] = []
        has_cands: List[List[bool]] = []

        for doc in docs:
            ents_cands.append(self.fetch_entity_info(doc)[0])
            # Determine which ents have candidates and should be included in prompt.
            has_cands.append(
                [
                    {cand_ent.id for cand_ent in cand_ents} != {EntityLinker.NIL}
                    or not self._auto_nil
                    for cand_ents in ents_cands[-1]
                ]
            )

        return ents_cands, has_cands

    def _get_prompt_data(
        self, shard: Doc, i_shard: int, i_doc: int, n_shards: int
    ) -> Dict[str, Any]:
        # n_shards changes before reset happens in _preprocess_docs() whenever sharding mechanism varies number of
        # shards. In this case we have to reset task state as well.
        if n_shards != self._n_shards:
            self._n_shards = n_shards
            self._ents_cands_by_shard = [[] * len(self._ents_cands_by_doc)]
            self._has_ent_cands_by_shard = [[] * len(self._ents_cands_by_doc)]

        # It's not ideal that we have to run candidate selection again here - but due to (1) us wanting to know whether
        # all entities have candidates before sharding and, more importantly, (2) some entities maybe being split up in
        # the sharding process it's cleaner to look for candidates again.
        if n_shards == 1:
            # If only one shard: shard is identical to original doc, so we don't have to rerun candidate search.
            ents_cands, has_cands = (
                self._ents_cands_by_doc[i_doc],
                self._has_ent_cands_by_doc[i_doc],
            )
        else:
            cands_info = self._find_entity_candidates([shard])
            ents_cands, has_cands = cands_info[0][0], cands_info[1][0]

        # Update shard-wise candidate info so it can be reused during parsing.
        if len(self._ents_cands_by_shard[i_doc]) == 0:
            self._ents_cands_by_shard[i_doc] = [[] for _ in range(n_shards)]
            self._has_ent_cands_by_shard[i_doc] = [[] for _ in range(n_shards)]
        self._ents_cands_by_shard[i_doc][i_shard] = ents_cands
        self._has_ent_cands_by_shard[i_doc][i_shard] = has_cands

        return {
            "mentions_str": ", ".join(
                [
                    f"*{mention.text}*"
                    for hc, mention in zip(has_cands, shard.ents)
                    if hc
                ]
            ),
            "mentions": [ent.text for hc, ent in zip(has_cands, shard.ents) if hc],
            "entity_descriptions": [
                [ent.description for ent in ents]
                for hc, ents in zip(has_cands, ents_cands)
                if hc
            ],
            "entity_ids": [
                [ent.id for ent in ents]
                for hc, ents in zip(has_cands, ents_cands)
                if hc
            ],
        }

    def parse_responses(
        self, shards: Iterable[Iterable[Doc]], responses: Iterable[Iterable[str]]
    ) -> Iterable[Doc]:
        shards_teed = self._tee_2d_iterable(shards, 2)
        parsed_responses = self._parse_responses(self, shards_teed[1], responses)

        for i_doc, (shards_for_doc, ent_spans_for_doc) in enumerate(
            zip(shards_teed[0], parsed_responses)
        ):
            updated_shards_for_doc: List[Doc] = []
            for i_shard, (shard, ent_spans) in enumerate(
                zip(shards_for_doc, ent_spans_for_doc)
            ):
                gen_nil_span: Callable[[Span], Span] = lambda ent: Span(  # noqa: E731
                    doc=shard,
                    start=ent.start,
                    end=ent.end,
                    label=ent.label,
                    vector=ent.vector,
                    vector_norm=ent.vector_norm,
                    kb_id=EntityLinker.NIL,
                )

                # If numbers of ents parsed from LLM response + ents without candidates and number of ents in doc don't
                # align, skip doc (most likely LLM parsing failed, no guarantee KB IDs can be assigned to correct ents).
                # This can happen when the LLM fails to list solutions for all entities.
                all_entities_resolved = len(ent_spans) + sum(
                    [
                        not is_in_prompt
                        for is_in_prompt in self._has_ent_cands_by_shard[i_doc][i_shard]
                    ]
                ) == len(shard.ents)

                # Fuse entities with (i. e. inferred by the LLM) and without candidates (i. e. auto-niled).
                # If entity was not included in prompt, as there were no candidates - fill in NIL for this entity.
                # If numbers of inferred and auto-niled entities don't line up with total number of entities, there is
                # no guaranteed way to assign a partially resolved list of entities
                # correctly.
                # Else: entity had candidates and was included in prompt - fill in resolved KB ID.
                ent_spans_iter = iter(ent_spans)
                shard.ents = [
                    gen_nil_span(ent)
                    if not (
                        all_entities_resolved
                        and self._has_ent_cands_by_shard[i_doc][i_shard][i_ent]
                    )
                    else next(ent_spans_iter)
                    for i_ent, ent in enumerate(shard.ents)
                ]

                # Remove entity highlights in shards.
                updated_shards_for_doc.append(
                    EntityLinkerTask.unhighlight_ents_in_doc(shard)
                )

            yield self._shard_reducer(self, updated_shards_for_doc)  # type: ignore[arg-type]

    def scorer(self, examples: Iterable[Example]) -> Dict[str, Any]:
        return self._scorer(examples)

    @property
    def _cfg_keys(self) -> List[str]:
        return ["_template"]

    @staticmethod
    def highlight_ents_in_doc(
        doc: Doc, include_ents: Optional[List[bool]] = None
    ) -> Doc:
        """Highlights entities in doc by wrapping them in **.
        doc (Doc): Doc whose entities are to be highlighted.
        include_ents (Optional[List[bool]]): Whether to include entities with the corresponding indices. If None, all
            are included.
        RETURNS (Doc): Doc with highlighted entities.
        """
        if include_ents is not None and len(include_ents) != len(doc.ents):
            raise ValueError(
                f"`include_ents` has {len(include_ents)} entries, but {len(doc.ents)} are required."
            )

        ents_to_highlight_idx = [
            i
            for i, ent in enumerate(doc.ents)
            if (include_ents is None or include_ents[i])
        ]
        ents_idx = [(ent.start, ent.end) for ent in doc.ents]

        # Include *-marker as tokens. Update entity indices.
        i_ent = 0
        new_ent_idx: List[Tuple[int, int]] = []
        token_texts: List[str] = []
        spaces: List[bool] = []
        to_highlight = i_ent in ents_to_highlight_idx
        offset = 0

        for token in doc:
            if i_ent < len(ents_idx) and token.i == ents_idx[i_ent][1]:
                if to_highlight:
                    token_texts.append("*")
                    spaces.append(spaces[-1])
                    spaces[-2] = False
                    offset += 1
                i_ent += 1
                to_highlight = i_ent in ents_to_highlight_idx
            if i_ent < len(ents_idx) and token.i == ents_idx[i_ent][0]:
                if to_highlight:
                    token_texts.append("*")
                    spaces.append(False)
                    offset += 1
                new_ent_idx.append(
                    (ents_idx[i_ent][0] + offset, ents_idx[i_ent][1] + offset)
                )
            token_texts.append(token.text)
            spaces.append(token.whitespace_ != "")

        # Cover edge case of doc ending with entity, in which case we need to close the * wrapping.
        if len(ents_to_highlight_idx) and doc.ents[
            ents_to_highlight_idx[-1]
        ].end == len(doc):
            token_texts.append("*")
            spaces.append(False)

        # Create doc with new tokens and entities.
        highlighted_doc = Doc(doc.vocab, words=token_texts, spaces=spaces)
        highlighted_doc.ents = [
            Span(
                doc=highlighted_doc,
                start=new_ent_idx[i][0],
                end=new_ent_idx[i][1],
                label=ent.label,
                vector=ent.vector,
                vector_norm=ent.vector_norm,
                kb_id=ent.kb_id_,
            )
            for i, ent in enumerate(doc.ents)
        ]

        return highlighted_doc

    @staticmethod
    def unhighlight_ents_in_doc(doc: Doc) -> Doc:
        """Remove entity highlighting (* wrapping)  in doc.
        doc (Doc): Doc whose entities are to be highlighted.
        RETURNS (Doc): Doc with highlighted entities.
        """
        highlight_start_idx = {
            ent.start - 1
            for ent in doc.ents
            if ent.start - 1 > 0 and doc[ent.start - 1].text == "*"
        }
        highlight_end_idx = {
            ent.end
            for ent in doc.ents
            if ent.end < len(doc) and doc[ent.end].text == "*"
        }
        highlight_idx = highlight_start_idx | highlight_end_idx

        # Compute entity indices with removed highlights.
        ent_idx: List[Tuple[int, int]] = []
        offset = 0
        for ent in doc.ents:
            is_highlighted = ent.start - 1 in highlight_start_idx
            ent_idx.append(
                (ent.start + offset - is_highlighted, ent.end + offset - is_highlighted)
            )
            offset -= 2 * is_highlighted

        # Create doc with new tokens and entities.
        tokens = [
            token
            for token in doc
            if not (token.i in highlight_idx and token.text == "*")
        ]
        unhighlighted_doc = Doc(
            doc.vocab,
            words=[token.text for token in tokens],
            # Use original token space, if token doesn't appear after * highlight. If so, insert space unconditionally.
            spaces=[
                token.whitespace_ != "" or token.i + 1 in highlight_idx
                for i, token in enumerate(tokens)
            ],
        )

        unhighlighted_doc.ents = [
            Span(
                doc=unhighlighted_doc,
                start=ent_idx[i][0],
                end=ent_idx[i][1],
                label=ent.label,
                vector=ent.vector,
                vector_norm=ent.vector_norm,
                kb_id=ent.kb_id_,
            )
            for i, ent in enumerate(doc.ents)
        ]

        return unhighlighted_doc

    def _require_candidate_selector(self) -> None:
        """Raises an error if candidate selector is not available."""
        if not self._candidate_selector:
            raise ValueError(
                "Candidate selector hasn't been initialized. Pass the corresponding config to "
                "[initialize.components.COMPONENT_NAME.candidate_selector]."
            )

    def fetch_entity_info(
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
    def has_ent_cands_by_shard(self) -> List[List[List[bool]]]:
        """Returns flags indicating whether shards' entities' have candidates in KB.
        RETURNS (List[List[List[bool]]]): Flags indicating whether shards' entities' have candidates in KB.
        """
        return self._has_ent_cands_by_shard
