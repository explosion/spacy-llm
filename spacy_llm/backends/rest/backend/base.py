import abc
import time
import warnings
from typing import Any, Dict, Iterable, Callable, Tuple

import requests  # type: ignore
from requests import ConnectTimeout, ReadTimeout


class Backend(abc.ABC):
    """Queries LLMs via their REST APIs."""

    def __init__(
        self,
        config: Dict[Any, Any],
        strict: bool,
        max_tries: int,
        interval: float,
        max_request_time: float,
    ):
        """Initializes new Backend instance.
        config (Dict[Any, Any]): Config passed on to LLM API.
        strict (bool): If True, ValueError is raised if the LLM API returns a malformed response (i. e. any kind of JSON
            or other response object that does not conform to the expectation of how a well-formed response object from
            this API should look like). If False, the API error responses are returned by __call__(), but no error will
            be raised.
            Note that only response object structure will be checked, not the prompt response text per se.
        max_tries (int): Max. number of tries for API request.
        interval (float): Time interval (in seconds) for API retries in seconds. We implement a base 2 exponential
            backoff at each retry.
        max_request_time (float): Max. time (in seconds) to wait for request to terminate before raising an exception.
        """
        self._config = config
        self._strict = strict
        self._max_tries = max_tries
        self._interval = interval
        self._max_request_time = max_request_time
        self._url = self._config.pop("url") if "url" in self._config else None
        self._credentials = self.credentials

        if "model" not in config:
            raise ValueError("The LLM model must be specified in the config.")
        self._check_api_endpoint_compatibility()

        assert self._max_tries >= 1
        assert self._interval > 0
        assert self._max_request_time > 0

    @property
    def _retry_error_codes(self) -> Tuple[int, ...]:
        """Returns codes qualifying as error (and triggering retrying requests).
        RETURNS (Tuple[int]): Error codes.
        """
        return 429, 503

    @abc.abstractmethod
    def __call__(self, prompts: Iterable[str]) -> Iterable[str]:
        """Executes prompts on specified API.
        prompts (Iterable[str]): Prompts to execute.
        RETURNS (Iterable[str]): API responses.
        """

    @property
    @abc.abstractmethod
    def supported_models(self) -> Dict[str, str]:
        """Returns supported models with their endpoints.
        RETURNS (Dict[str, str]): Supported models with their endpoints.
        """

    @property
    @abc.abstractmethod
    def credentials(self) -> Dict[str, str]:
        """Get credentials for the LLM API.
        RETURNS (Dict[str, str]): Credentials.
        """

    def retry(
        self, call_method: Callable[..., requests.Response], url: str, **kwargs
    ) -> requests.Response:
        """Retry a call to an API if we get a non-ok status code.
        This function automatically retries a request if it catches a response with an error code in `error_codes`.
        The time interval also increases exponentially every time we retry.
        call_method (Callable[[str, ...], requests.Response]): Method to use to fetch request. Must accept URL as first
            parameter.
        url (str): URL to address in request.
        kwargs: Keyword args to be passed on to request.
        RETURNS (requests.Response): Response of last call.
        """

        def _call_api() -> requests.Response:
            """Calls API with given timeout.
            RETURNS (requests.Response): Response object.
            """
            try:
                return call_method(url, **kwargs)
            except (ConnectTimeout, ReadTimeout, TimeoutError) as err:
                raise TimeoutError(
                    "Request time out. Check your network connection and the API's availability."
                ) from err

        interval = self._interval
        response = _call_api()
        i = 0

        # We don't want to retry on every non-ok status code. Some are about
        # incorrect inputs, etc. and we want to terminate on those.
        start_time = time.time()
        while i < self._max_tries and response.status_code in self._retry_error_codes:
            time.sleep(interval)
            response = _call_api()
            i += 1
            # Increase timeout everytime you retry
            interval = interval * 2

        if response.status_code in self._retry_error_codes:
            raise ConnectionError(
                f"API could not be reached after {(time.time() - start_time):.3f} seconds in total and attempting to "
                f"connect {self._max_tries} times. Check your network connection and the API's availability."
            )

        return response

    def _check_api_endpoint_compatibility(self):
        """Checks whether specified model supports the supported API endpoint."""
        supported_models = self.supported_models
        if self._config["model"] not in supported_models:
            raise ValueError(
                f"Requested model '{self._config['model']}' is not one of the supported models: "
                f"{', '.join(sorted(list(supported_models.keys())))}."
            )

        model_endpoint = supported_models[self._config["model"]]
        if self._url and self._url != model_endpoint:
            warnings.warn(
                f"Configured endpoint {self._url} diverges from expected endpoint {model_endpoint} for selected "
                f"model '{self._config['model']}'. Please ensure that this endpoint supports your model."
            )
