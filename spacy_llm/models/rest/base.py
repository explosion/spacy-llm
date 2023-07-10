import abc
import time
from enum import Enum
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import requests  # type: ignore
from requests import ConnectTimeout, ReadTimeout


class _HTTPRetryErrorCodes(Enum):
    TOO_MANY_REQUESTS = 429
    SERVICE_UNAVAILABLE = 503

    @classmethod
    def has(cls, item: int):
        return item in set(item.value for item in cls)


class REST(abc.ABC):
    """Queries LLMs via their REST APIs."""

    DEFAULT_STRICT = True
    DEFAULT_MAX_TRIES = 5
    DEFAULT_INTERVAL = 1.0
    DEFAULT_MAX_REQUEST_TIME = 30

    def __init__(
        self,
        name: str,
        endpoint: str,
        config: Dict[Any, Any],
        strict: bool,
        max_tries: int,
        interval: float,
        max_request_time: float,
    ):
        """Initializes new instance of REST-based model.
        name (str): Model name.
        endpoint (str): URL of API endpoint.
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
        self._name = name
        self._endpoint = endpoint
        self._config = config
        self._strict = strict
        self._max_tries = max_tries
        self._interval = interval
        self._max_request_time = max_request_time
        self._credentials = self.credentials

        assert self._max_tries >= 1
        assert self._interval > 0
        assert self._max_request_time > 0

        self._check_model()
        self._verify_auth()

    def _check_model(self) -> None:
        """Checks whether model is supported. Raises if it isn't."""
        if self._name not in self.get_model_names():
            raise ValueError(
                f"Model '{self._name}' is not supported - select one of {self.get_model_names()} instead"
            )

    @abc.abstractmethod
    def __call__(self, prompts: Iterable[str]) -> Iterable[str]:
        """Executes prompts on specified API.
        prompts (Iterable[str]): Prompts to execute.
        RETURNS (Iterable[str]): API responses.
        """

    @classmethod
    @abc.abstractmethod
    def get_model_names(cls) -> Tuple[str, ...]:
        """Names of supported models.
        RETURNS (Tuple[str]): Names of supported models.
        """

    @property
    @abc.abstractmethod
    def credentials(self) -> Dict[str, str]:
        """Get credentials for the LLM API.
        RETURNS (Dict[str, str]): Credentials.
        """

    @abc.abstractmethod
    def _verify_auth(self) -> None:
        """Verifiy API authentication (and model choice, if possible)."""

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

        def _call_api(attempt: int) -> Optional[requests.Response]:
            """Calls API with given timeout.
            attempt (int): Reflects the how many-th try at reaching the API this is. If attempt < self._max_tries and
                the call fails, None is returned. If attempt == self._max_tries and the call fails, a TimeoutError is
                raised.
            RETURNS (Optional[requests.Response]): Response object.
            """
            try:
                return call_method(url, **kwargs)
            except (ConnectTimeout, ReadTimeout, TimeoutError) as err:
                if attempt < self._max_tries:
                    return None
                else:
                    raise TimeoutError(
                        "Request time out. Check your network connection and the API's availability."
                    ) from err

        interval = self._interval
        i = 0
        response = _call_api(i)

        # We don't want to retry on every non-ok status code. Some are about
        # incorrect inputs, etc. and we want to terminate on those.
        start_time = time.time()
        while i < self._max_tries and (
            response is None or _HTTPRetryErrorCodes.has(response.status_code)
        ):
            time.sleep(interval)
            response = _call_api(i + 1)
            i += 1
            # Increase timeout everytime you retry
            interval = interval * 2

        assert isinstance(response, requests.Response)
        if _HTTPRetryErrorCodes.has(response.status_code):
            raise ConnectionError(
                f"API could not be reached after {(time.time() - start_time):.3f} seconds in total and attempting to "
                f"connect {self._max_tries} times. Check your network connection and the API's availability.\n"
                f"{response.status_code}\t{response.reason}"
            )

        return response
