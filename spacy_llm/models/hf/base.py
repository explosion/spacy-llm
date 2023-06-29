import abc
import warnings
from typing import Any, Dict, Iterable, Optional, Tuple

from thinc.compat import has_torch_cuda_gpu

from ...compat import Literal, has_accelerate, has_torch, has_transformers, torch


class HuggingFace(abc.ABC):
    """Base class for HuggingFace model classes."""

    MODEL_NAMES = Literal[None]  # noqa: F722

    def __init__(
        self,
        name: str,
        config_init: Optional[Dict[str, Any]],
        config_run: Optional[Dict[str, Any]],
    ):
        """Initializes HF model instance.
        query (Callable[[Any, Iterable[Any]], Iterable[Any]): Callable executing LLM prompts when
            supplied with the `integration` object.
        name (str): Name of HF model to load (without account name).
        config_init (Optional[Dict[str, Any]]): HF config for initializing the model.
        config_run (Optional[Dict[str, Any]]): HF config for running the model.
        inference_config (Dict[Any, Any]): HF config for model run.
        """
        self._name = name if self.hf_account in name else f"{self.hf_account}/{name}"
        self._config_init, self._config_run = self.compile_default_configs()
        if config_init:
            self._config_init = {**self._config_init, **config_init}
        if config_run:
            self._config_run = {**self._config_run, **config_run}

        # Init HF model.
        HuggingFace.check_installation()
        self._check_model()
        self._model = self.init_model()

    @abc.abstractmethod
    def __call__(self, prompts: Iterable[Any]) -> Iterable[Any]:
        """Executes prompts on specified API.
        prompts (Iterable[Any]): Prompts to execute.
        RETURNS (Iterable[Any]): API responses.
        """

    def _check_model(self) -> None:
        """Checks whether model is supported. Raises if it isn't."""
        if self._name.replace(f"{self.hf_account}/", "") not in self.get_model_names():
            raise ValueError(
                f"Model '{self._name}' is not supported - select one of {self.get_model_names()} instead"
            )

    @classmethod
    def get_model_names(cls) -> Tuple[str, ...]:
        """Names of supported models for this HF model implementation.
        RETURNS (Tuple[str]): Names of supported models.
        """
        return tuple(str(arg) for arg in cls.MODEL_NAMES.__args__)  # type: ignore[attr-defined]

    @property
    @abc.abstractmethod
    def hf_account(self) -> str:
        """Name of HF account for this model.
        RETURNS (str): Name of HF account.
        """

    @staticmethod
    def check_installation() -> None:
        """Checks whether the required external libraries are installed. Raises an error otherwise."""
        if not has_torch:
            raise ValueError(
                "The HF model requires `torch` to be installed, which it is not. See "
                "https://pytorch.org/ for installation instructions."
            )
        if not has_transformers:
            raise ValueError(
                "The HF model requires `transformers` to be installed, which it is not. See "
                "https://huggingface.co/docs/transformers/installation for installation instructions."
            )

    @staticmethod
    def compile_default_configs() -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Compiles default init and run configs for HF model.
        RETURNS (Tuple[Dict[str, Any], Dict[str, Any]]): HF model default init config, HF model default run config.
        """
        default_cfg_init: Dict[str, Any] = {}
        default_cfg_run: Dict[str, Any] = {}

        if has_torch:
            default_cfg_init["torch_dtype"] = torch.bfloat16
            if has_torch_cuda_gpu:
                # this ensures it fails explicitely when GPU is not enabled or sufficient
                default_cfg_init["device"] = "cuda:0"
            elif has_accelerate:
                # accelerate will distribute the layers depending on availability on GPU/CPU/hard drive
                default_cfg_init["device_map"] = "auto"
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
        return default_cfg_init, default_cfg_run

    @abc.abstractmethod
    def init_model(self) -> Any:
        """Sets up HF model and needed utilities.
        RETURNS (Any): HF model.
        """
