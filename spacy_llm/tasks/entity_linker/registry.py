import warnings
from pathlib import Path
from typing import Optional, Type, Union

from spacy.scorer import Scorer

from ...registry import registry
from ...ty import ExamplesConfigType, FewshotExample, TaskResponseParser
from .candidate_selector import PipelineCandidateSelector
from .parser import parse_responses_v1
from .task import DEFAULT_EL_TEMPLATE_V1, EntityLinkerTask
from .ty import EntDescReader, InMemoryLookupKBLoader
from .util import ELExample, KBSerializedLoader, KBYamlLoader, ent_desc_reader_csv
from .util import score


@registry.llm_tasks("spacy.EntityLinker.v1")
def make_entitylinker_task(
    template: str = DEFAULT_EL_TEMPLATE_V1,
    parse_responses: Optional[TaskResponseParser[EntityLinkerTask]] = None,
    prompt_example_type: Optional[Type[FewshotExample]] = None,
    examples: ExamplesConfigType = None,
    scorer: Optional[Scorer] = None,
):
    """EntityLinker.v1 task factory.

    template (str): Prompt template passed to the model.
    parse_responses (Optional[TaskResponseParser]): Callable for parsing LLM responses for this task.
    prompt_example_type (Optional[Type[FewshotExample]]): Type to use for fewshot examples.
    examples (ExamplesConfigType): Optional callable that reads a file containing task examples for few-shot learning.
        If None is passed, then zero-shot learning will be used.
    scorer (Optional[Scorer]): Scorer function.
    """
    raw_examples = examples() if callable(examples) else examples
    example_type = prompt_example_type or ELExample
    examples = [example_type(**eg) for eg in raw_examples] if raw_examples else None
    # Ensure there is a reason for every solution, even if it's empty. This makes templating easier.
    if examples:
        for example in examples:
            if example.reasons is None:
                example.reasons = [""] * len(example.solutions)
            elif 0 < len(example.reasons) < len(example.solutions):
                warnings.warn(
                    f"The number of reasons doesn't match the number of solutions ({len(example.reasons)} "
                    f"vs. {len(example.solutions)}). There must be one reason per solution for an entity "
                    f"linking example, or no reasons at all. Ignoring all specified reasons."
                )
                example.reasons = [""] * len(example.solutions)

    return EntityLinkerTask(
        template=template,
        parse_responses=parse_responses or parse_responses_v1,
        prompt_example_type=example_type,
        prompt_examples=examples,
        scorer=scorer or score,
    )


@registry.llm_misc("spacy.CandidateSelector.v1")
def make_candidate_selector_pipeline(
    kb_loader: InMemoryLookupKBLoader,
    top_n: int = 5,
) -> PipelineCandidateSelector:
    """Generates CandidateSelector. Note that this class has to be initialized (.initialize()) before being used.
    kb_loader (InMemoryLookupKBLoader): KB loader.
    top_n (int): Top n candidates to include in prompt.
    """
    # Note: we could also move the class implementation here directly. This was just done to separate registration from
    # implementation code.
    return PipelineCandidateSelector(
        kb_loader=kb_loader,
        top_n=top_n,
    )


@registry.llm_misc("spacy.EntityDescriptionReader.v1")
def make_ent_desc_reader() -> EntDescReader:
    """Instantiates entity description reader with two columns: ID and description.
    RETURNS (Dict[str, str]): Dict with ID -> description.
    """
    return ent_desc_reader_csv


@registry.llm_misc("spacy.KBSerializedLoader.v1")
def make_kb_serialized_loader(
    path: Union[str, Path],
    nlp_path: Optional[Union[str, Path]] = None,
    desc_path: Optional[Union[str, Path]] = None,
    ent_desc_reader: Optional[EntDescReader] = None,
) -> KBSerializedLoader:
    """Instantiates KBSerializedLoader for loading KBs from serialized directories (as done during spaCy pipeline
    serialization).
    path (Union[str, Path]): Path to KB directory.
    nlp_path (Optional[Union[str, Path]]): Path to NLP pipeline whose vocab data to use. If this is None, the loader
        will try to load the serialized pipeline surrounding the KB directory.
    desc_path (Optional[Union[Path, str]]): Path to .csv file with descriptions for entities. Has to have two
        columns with the first one being the entity ID, the second one being the description. The entity ID has to
        match with the entity ID in the stored knowledge base.
        If not specified, all entity descriptions provided in prompts will be a generic "No description available"
        or something else to this effect.
    ent_desc_reader (EntDescReader): Entity description reader.
    RETURNS (KBSerializedLoader): Loader instance.
    """
    return KBSerializedLoader(
        path=path,
        nlp_path=nlp_path,
        desc_path=desc_path,
        ent_desc_reader=ent_desc_reader or ent_desc_reader_csv,
    )


@registry.llm_misc("spacy.KBYamlLoader.v1")
def make_kb_yaml_loader(path: Union[str, Path]) -> KBYamlLoader:
    """Instantiates KBYamlLoader for generating KBs from .yaml file containing entity data.
    path (Union[str, Path]): Path to KB directory.
    RETURNS (KBYamlLoader): Loader instance.
    """
    return KBYamlLoader(path=path)
