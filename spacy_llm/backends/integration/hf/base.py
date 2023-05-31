import abc
import warnings
from typing import Iterable, TypeVar, Any, Dict
from ....compat import has_torch, has_transformers, has_accelerate, torch
from thinc.compat import has_torch_cuda_gpu

# Type of prompts returned from Task.generate_prompts().
_PromptType = TypeVar("_PromptType")
# Type of responses returned from query function.
_ResponseType = TypeVar("_ResponseType")


class HuggingFaceBackend(abc.ABC):
    """Backend for HuggingFace models."""

    def __init__(
        self,
        model: str,
        config: Dict[str, Dict[Any, Any]],
    ):
        """Initializes Backend instance.
        query (Callable[[Any, Iterable[_PromptType]], Iterable[_ResponseType]]): Callable executing LLM prompts when
            supplied with the `integration` object.
        model (str): Name of HF model to load.
        config (Dict[Any, Any]): HF config. Has to have keys "init" for all arguments used for model at init time and
            "run" for all arguments at inference time.
        inference_config (Dict[Any, Any]): HF config for model run.
        """
        self._model_name = model
        self._config = config
        assert len(self._config) == 0 or all([key in ("init", "run") for key in config])

        # Init HF model.
        HuggingFaceBackend.check_installation()
        self.check_model()

        if not self._config or len(self._config) == 0:
            self._config = self.compile_default_config()

        self._model = self.init_model()

    @abc.abstractmethod
    def __call__(self, prompts: Iterable[_PromptType]) -> Iterable[_ResponseType]:
        """Executes prompts on specified API.
        prompts (Iterable[_PromptType]): Prompts to execute.
        RETURNS (Iterable[__ResponseType]): API responses.
        """

    @property
    @abc.abstractmethod
    def supported_models(self) -> Iterable[str]:
        """Get names of supported models.
        RETURNS (Iterable[str]): Names of supported models.
        """

    @staticmethod
    def check_installation() -> None:
        """Checks whether the required external libraries are installed. Raises an error otherwise."""
        if not has_torch:
            raise ValueError(
                "The HF backend requires `torch` to be installed, which it is not. See "
                "https://pytorch.org/ for installation instructions."
            )
        if not has_transformers:
            raise ValueError(
                "The HF backend requires `transformers` to be installed, which it is not. See "
                "https://huggingface.co/docs/transformers/installation for installation instructions."
            )

    def check_model(self) -> None:
        """Checks whether model is supported. Raises if it isn't."""
        if self._model_name not in self.supported_models:
            raise ValueError(
                f"Model '{self._model_name}' is not supported - select one of {self.supported_models} instead"
            )

    @staticmethod
    def compile_default_config() -> Dict[Any, Any]:
        """Compiles default config for HF model.
        RETURNS (Dict[Any, Any]): HF model default config.
        """
        default_cfg: Dict[str, Any] = {"init": {}, "run": {}}

        if has_torch:
            default_cfg["init"]["torch_dtype"] = torch.bfloat16
            if has_torch_cuda_gpu:
                # this ensures it fails explicitely when GPU is not enabled or sufficient
                default_cfg["init"]["device"] = "cuda:0"
            elif has_accelerate:
                # accelerate will distribute the layers depending on availability on GPU/CPU/hard drive
                default_cfg["init"]["device_map"] = "auto"
                warnings.warn(
                    "Couldn't find a CUDA GPU, so the setting 'device_map:auto' will be used, which may result "
                    "in the LLM being loaded (partly) on the CPU or even the hard disk, which may be slow. "
                    "Install cuda to be able to load and run the LLM on the GPU instead."
                )
            else:
                raise ValueError(
                    "Install CUDA to load and run the LLM on the GPU, or install 'accelerate' to dynamically "
                    "distribute the LLM on the CPU or even the hard disk. The latter may be slow."
                )
        return default_cfg

    @abc.abstractmethod
    def init_model(self) -> Any:
        """Sets up HF model and needed utilities.
        RETURNS (Any): HF model.
        """
