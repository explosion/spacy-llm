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
        context_length: Optional[int],
    ):
        """Initializes HF model instance.
        query (Callable[[Any, Iterable[Any]], Iterable[Any]): Callable executing LLM prompts when
            supplied with the `integration` object.
        name (str): Name of HF model to load (without account name).
        config_init (Optional[Dict[str, Any]]): HF config for initializing the model.
        config_run (Optional[Dict[str, Any]]): HF config for running the model.
        context_length (Optional[int]): Context length for this model. Necessary for sharding.
        """
        self._name = name if self.hf_account in name else f"{self.hf_account}/{name}"
        self._context_length = context_length
        default_cfg_init, default_cfg_run = self.compile_default_configs()
        self._config_init, self._config_run = default_cfg_init, default_cfg_run

        if config_init:
            self._config_init = {**self._config_init, **config_init}
        if config_run:
            self._config_run = {**self._config_run, **config_run}

        # `device` and `device_map` are conflicting arguments - ensure they aren't both set.
        if config_init:
            # Case 1: both device and device_map explicitly set by user.
            if "device" in config_init and "device_map" in config_init:
                warnings.warn(
                    "`device` and `device_map` are conflicting arguments - don't set both. Dropping argument "
                    "`device`."
                )
                self._config_init.pop("device")
            # Case 2: we have a CUDA GPU (and hence device="cuda:0" by default), but device_map is set by user.
            elif "device" in default_cfg_init and "device_map" in config_init:
                self._config_init.pop("device")
            # Case 3: we don't have a CUDA GPU (and hence "device_map=auto" by default), but device is set by user.
            elif "device_map" in default_cfg_init and "device" in config_init:
                self._config_init.pop("device_map")

        # Fetch proper torch.dtype, if specified.
        if (
            has_torch
            and "torch_dtype" in self._config_init
            and self._config_init["torch_dtype"] != "auto"
        ):
            try:
                self._config_init["torch_dtype"] = getattr(
                    torch, self._config_init["torch_dtype"]
                )
            except AttributeError as ex:
                raise ValueError(
                    f"Invalid value {self._config_init['torch_dtype']} was specified for `torch_dtype`. "
                    f"Double-check you specified a valid dtype."
                ) from ex

        # Init HF model.
        HuggingFace.check_installation()
        self._check_model()
        self._model = self.init_model()

    @abc.abstractmethod
    def __call__(self, prompts: Iterable[Iterable[Any]]) -> Iterable[Iterable[Any]]:
        """Executes prompts on specified API.
        prompts (Iterable[Iterable[Any]]): Prompts to execute per doc.
        RETURNS (Iterable[Iterable[Any]]): API responses per doc.
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
    def context_length(self) -> Optional[int]:
        """Returns context length in number of tokens for this model.
        RETURNS (Optional[int]): Max. number of tokens allowed in prompt for the current model.
        """
        return self._context_length

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
            default_cfg_init["torch_dtype"] = "bfloat16"
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
