import csv
import re
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, Union

import jinja2
import spacy
from pydantic import BaseModel
from spacy.language import Language
from spacy.pipeline import EntityLinker
from spacy.scorer import Scorer
from spacy.tokens import Doc, Span
from spacy.training import Example

from ..registry import registry
from ..ty import CandidateSelector, ExamplesConfigType
from .templates import read_template
from .util import SerializableTask

_DEFAULT_EL_TEMPLATE_V1 = read_template("entity_linking")


class EntityLinkingExample(BaseModel):
    text: str
    mentions_str: str
    mentions: List[str]
    entity_descriptions: List[List[str]]
    entity_ids: List[List[str]]
    solutions: List[str]
    reasons: Optional[List[str]]


@registry.llm_misc("spacy.CandidateSelectorPipeline.v1")
class SpaCyPipelineCandidateSelector:
    """Callable generated by loading and wrapping a spaCy pipeline with a NEL component and a filled knowledge base."""

    def __init__(
        self,
        nlp_path: Union[Path, str],
        desc_path: Union[Path, str],
        top_n: int = 5,
    ):
        """
        Loads spaCy pipeline, knowledge base, entity descriptions.
        top_n (int): Top n candidates to include in prompt.
        nlp_path (Union[Path, str]): Path to stored spaCy pipeline.
        desc_path (Union[Path, str]): Path to .csv file with descriptions for entities. Has to have two columns
          with the first one being the entity ID, the second one being the description. The entity ID has to match with
          the entity ID in the stored knowledge base.
        """
        self._nlp = spacy.load(nlp_path)
        if "entity_linker" not in self._nlp.component_names:
            raise ValueError(
                f"'entity_linker' component has to be available in specified pipeline at {nlp_path}, but "
                f"isn't."
            )
        self._entity_linker: EntityLinker = self._nlp.get_pipe("entity_linker")
        self._kb = self._entity_linker.kb
        with open(desc_path) as csvfile:
            self._descs = {}
            for row in csv.reader(csvfile, quoting=csv.QUOTE_ALL, delimiter=";"):
                try:
                    self._descs[row[0]] = row[1]
                except IndexError as ex:  # noqa: F841
                    print(row)  # noqa: T201
                    continue
            # self._descs = {
            #     row[0]: row[1]
            #     for row in csv.reader(csvfile, quoting=csv.QUOTE_ALL, delimiter=";")
            # }
        self._top_n = top_n

    def __call__(self, mentions: Iterable[Span]) -> Iterable[Dict[str, str]]:
        """Retrieves top n candidates using spaCy's entity linker's .get_candidates_batch().
        mentions (Iterable[Span]): Mentions to look up entity candidates for.
        RETURNS (Iterable[Dict[str, str]]): Dicts of entity ID -> description for all candidates, per mention.
        """
        all_cands = self._kb.get_candidates_batch(mentions)
        for cands in all_cands:
            assert isinstance(cands, list)
            cands.sort(key=lambda x: x.prior_prob, reverse=True)

        return [
            {cand.entity_: self._descs[cand.entity_] for cand in cands[: self._top_n]}
            for cands in all_cands
        ]

    def get_entity_description(self, entity_id: str) -> str:
        if entity_id not in self._descs:
            raise ValueError(
                f"Entity with ID {entity_id} is not in provided descriptions file."
            )

        return self._descs[entity_id]


@registry.llm_tasks("spacy.EntityLinking.v1")
def make_entitylinking_task(
    candidate_selector: CandidateSelector,
    template: str = _DEFAULT_EL_TEMPLATE_V1,
    examples: ExamplesConfigType = None,
):
    """EntityLinking.v1 task factory.

    template (str): Prompt template passed to the model.
    examples (Optional[Callable[[], Iterable[Any]]]): Optional callable that
        reads a file containing task examples for few-shot learning. If None is
        passed, then zero-shot learning will be used.
    candidate_selector (CandidateSelector): Factory for a candidate selection callable
        returning candidates for a given Span and context.
    """
    raw_examples = examples() if callable(examples) else examples
    examples = (
        [EntityLinkingExample(**eg) for eg in raw_examples] if raw_examples else None
    )
    # Ensure there is a reason for every solution, even if it's empty. This makes templating easier.
    if examples:
        for example in examples:
            if example.reasons is None:
                example.reasons = [""] * len(example.solutions)
            elif len(example.reasons) < len(example.solutions):
                example.reasons.extend(
                    [""] * (len(example.solutions) - len(example.reasons))
                )

    return EntityLinkingTask(
        template=template, examples=examples, candidate_selector=candidate_selector
    )


class EntityLinkingTask(SerializableTask[EntityLinkingExample]):
    def __init__(
        self,
        candidate_selector: CandidateSelector,
        template: str = _DEFAULT_EL_TEMPLATE_V1,
        examples: Optional[List[EntityLinkingExample]] = None,
    ):
        """Default entity linking task.

        template (str): Prompt template passed to the model.
        examples (Optional[Callable[[], Iterable[Any]]]): Optional callable that
            reads a file containing task examples for few-shot learning. If None is
            passed, then zero-shot learning will be used.
        candidate_selector (CandidateSelector): Factory for a candidate selection callable
            returning candidates for a given Span and context.
        """
        self._template = template
        self._prompt_examples = examples or []
        self._candidate_selector = candidate_selector

    def initialize(
        self,
        get_examples: Callable[[], Iterable["Example"]],
        nlp: Language,
        n_prompt_examples: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initializes prompt examples from Doc examples.
        get_examples (Callable[[], Iterable["Example"]]): Callable that provides examples
            for initialization.
        nlp (Language): Language instance.
        n_prompt_examples (int): How many prompt examples to infer from the provided Example objects.
            0 by default. Takes all examples if set to -1.
        """
        for eg in get_examples():
            if n_prompt_examples < 0 or len(self._prompt_examples) < n_prompt_examples:
                prompt_example = self._create_prompt_example(eg)
                if prompt_example:
                    self._prompt_examples.append(prompt_example)

    @property
    def prompt_template(self) -> str:
        return self._template

    def generate_prompts(self, docs: Iterable[Doc]) -> Iterable[str]:
        environment = jinja2.Environment()
        _template = environment.from_string(self._template)
        for doc in docs:
            cands_ents_info, _ = self._fetch_entity_info(doc)
            yield _template.render(
                text=EntityLinkingTask._highlight_ents_in_text(doc),
                mentions_str=", ".join([f"*{mention}*" for mention in doc.ents]),
                mentions=[ent.text for ent in doc.ents],
                entity_descriptions=[
                    cands_ent_info[1] for cands_ent_info in cands_ents_info
                ],
                entity_ids=[cands_ent_info[0] for cands_ent_info in cands_ents_info],
                examples=self._prompt_examples,
            )

    def parse_responses(
        self, docs: Iterable[Doc], responses: Iterable[str]
    ) -> Iterable[Doc]:
        for doc, prompt_response in zip(docs, responses):
            solutions = [
                sol.replace("--- ", "").replace("<", "").replace(">", "")
                for sol in re.findall(r"--- <.*>", prompt_response)
            ]
            # Skip document if the numbers of entities and solutions don't line up.
            if len(solutions) != len(doc.ents):
                yield doc

            # Set ents anew by copying them and specifying the KB ID.
            doc.ents = [
                Span(
                    doc=doc,
                    start=ent.start,
                    end=ent.end,
                    label=ent.label,
                    vector=ent.vector,
                    vector_norm=ent.vector_norm,
                    kb_id=solution,
                )
                for ent, solution in zip(doc.ents, solutions)
            ]

            yield doc

    def scorer(
        self,
        examples: Iterable[Example],
    ) -> Dict[str, Any]:
        """Scores lemmatization accuracy on provided examples.
        examples (Iterable[Example]): Examples to determine score against.
        """
        return Scorer.score_links(examples, negative_labels=[EntityLinker.NIL])

    @property
    def _cfg_keys(self) -> List[str]:
        return ["_template"]

    @property
    def _Example(self) -> Type[EntityLinkingExample]:
        return EntityLinkingExample

    def _create_prompt_example(
        self, example: Example
    ) -> Optional[EntityLinkingExample]:
        """Create a entity linking prompt example from a spaCy example."""
        # Ensure that all entities have their knowledge base IDs set.
        n_ents = len(example.reference.ents)
        n_set_kb_ids = sum([ent.kb_id != 0 for ent in example.reference.ents])
        if n_ents and n_ents != n_set_kb_ids:
            warnings.warn(
                f"Not all entities in this document have their knowledge base IDs set ({n_set_kb_ids} out of "
                f"{n_ents}). Ignoring example:\n{example.reference}"
            )
            return None

        # Assemble example.
        mentions = [ent.text for ent in example.reference.ents]
        # Fetch candidates. If true entity not among candidates: fetch description separately and add manually.
        cands_ents_info, solutions = self._fetch_entity_info(example.reference)
        # If we are to use available docs as examples, they have to have KB IDs set and hence available solutions.
        assert all([sol is not None for sol in solutions])

        return EntityLinkingExample(
            text=EntityLinkingTask._highlight_ents_in_text(example.reference),
            mentions_str=", ".join([f"*{mention}*" for mention in mentions]),
            mentions=mentions,
            entity_descriptions=[
                cands_ent_info[1] for cands_ent_info in cands_ents_info
            ],
            entity_ids=[cands_ent_info[0] for cands_ent_info in cands_ents_info],
            solutions=solutions,
            reasons=[""] * len(mentions),
        )

    @staticmethod
    def _highlight_ents_in_text(doc: Doc) -> str:
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

        return text

    def _fetch_entity_info(
        self, doc: Doc
    ) -> Tuple[List[Tuple[List[str], List[str]]], List[Optional[str]]]:
        """Fetches entity IDs & descriptions and determines solution numbers for entities in doc.
        doc (Doc): Doc to fetch entity descriptions and solution numbers for. If entities' KB IDs are not set,
            corresponding solution number will be None.
        RETURNS (Tuple[List[Tuple[List[str], List[str]]], List[str]]): For each mention in doc: list of candidates
            with ID and description, list of correct entity IDs.
        """
        cands_per_ent = self._candidate_selector(doc.ents)
        cand_entity_info: List[Tuple[List[str], List[str]]] = []
        correct_ent_ids: List[Optional[str]] = []

        for ent, cands in zip(doc.ents, cands_per_ent):
            cand_ent_ids = list(cands.keys())
            cand_ent_descs: List[str] = [cands[ent_id] for ent_id in cand_ent_ids]

            # No KB ID known: In this case there is no guarantee that the correct entity description will be included.
            if ent.kb_id == 0:
                correct_ent_ids.append(None)
            # Correct entity not in suggested candidates: fetch description explicitly.
            elif ent.kb_id not in cands:
                cand_ent_descs.append(
                    self._candidate_selector.get_entity_description(ent.kb_id_)
                )
                cand_ent_ids.append(ent.kb_id)
            correct_ent_ids.append(ent.kb_id)

            cand_entity_info.append((cand_ent_ids, cand_ent_descs))

        return cand_entity_info, correct_ent_ids
