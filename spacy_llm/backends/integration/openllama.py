from typing import Iterable, Callable, Any, Dict, Optional

from spacy.util import SimpleFrozenList, SimpleFrozenDict

from . import HuggingFaceBackend
from ...compat import transformers
from ...registry.util import registry


class OpenLLaMaHFBackend(HuggingFaceBackend):
    tokenizer: Optional["transformers.AutoTokenizer"] = None

    def init_model(self) -> "transformers.AutoModelForCausalLM":
        """Sets up HF model and needed utilities.
        RETURNS (Any): HF model.
        """
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model, use_fast=False
        )
        return transformers.AutoModelForCausalLM.from_pretrained(
            self.model, **self.config
        )

    def __call__(self, prompts: Iterable[str]) -> Iterable[str]:  # type: ignore[override]
        assert self.tokenizer is not None
        tokenized_prompts = [
            {
                k: v.cuda()
                for k, v in self.tokenizer(
                    pr,
                    return_tensors="pt",
                    return_attention_mask=False,
                    add_special_tokens=False,
                ).items()
            }
            for pr in prompts
        ]

        return self.query(self.integration, tokenized_prompts)

    @property
    def supported_models(self) -> Iterable[str]:
        return SimpleFrozenList(["s-JoL/Open-Llama-V2-pretrain"])


def query_openllama(
    pipeline: "transformers.pipeline", prompts: Iterable[str]
) -> Iterable[str]:
    """Queries OpenLLaMa HF model.
    pipeline (transformers.pipeline): Transformers pipeline to query.
    prompts (Iterable[str]): Prompts to query Dolly model with.
    RETURNS (Iterable[str]): Prompt responses.
    """
    # for pr in prompts:
    #     inputs = self.tokenizer(
    #         pr,
    #         return_tensors="pt",
    #         return_attention_mask=False,
    #         add_special_tokens=False,
    #     )
    #     for k, v in inputs.items():
    #         inputs[k] = v.cuda()
    # pred = model.generate(**inputs, max_new_tokens=512, do_sample=True)
    # print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))

    return [pipeline(pr)[0]["generated_text"] for pr in prompts]


@registry.llm_backends("spacy.OpenLLaMaHF.v1")
def backend_openllama_hf(
    model: str,
    config: Dict[Any, Any] = SimpleFrozenDict(),
) -> Callable[[Iterable[str]], Iterable[str]]:
    """Returns Callable that can execute a set of prompts and return the raw responses.
    model (str): Name of the HF model.
    config (Dict[Any, Any]): config arguments passed on to the initialization of transformers.pipeline instance.
    RETURNS (Callable[[Iterable[str]], Iterable[str]]): Callable executing the prompts and returning raw responses.
    """
    return OpenLLaMaHFBackend(
        integration=None,
        query=query_openllama,  # type: ignore[arg-type]
        model=model,
        config=config,
    )
