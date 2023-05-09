import abc
import os
import time
from typing import Any, Dict, Iterable, List, Callable, Tuple, Set

import requests  # type: ignore
import srsly  # type: ignore[import]


class Backend(abc.ABC):
    """Queries LLMs via their REST APIs."""

    def __init__(
        self,
        config: Dict[Any, Any],
        strict: bool,
        max_tries: int,
        timeout: int,
    ):
        """Initializes new Backend instance.
        config (Dict[Any, Any]): Config passed on to LLM API.
        strict (bool): If True, ValueError is raised if the LLM API returns a malformed response (i. e. any kind of JSON
            or other response object that does not conform to the expectation of how a well-formed response object from
            this API should look like). If False, the API error responses are returned by __call__(), but no error will
            be raised.
            Note that only response object structure will be checked, not the prompt response text per se.
        max_tries (int): Max. number of tries for API request.
        timeout (int): Timeout for API request in seconds.
        """
        self._config = config
        self._strict = strict
        self._max_tries = max_tries
        self._timeout = timeout
        self._url = (
            self._config.pop("url")
            if "url" in self._config
            else self._default_endpoint
        )
        self._credentials = self.credentials

        assert self._max_tries >= 1
        assert self._timeout >= 1

    @property
    @abc.abstractmethod
    def _default_endpoint(self) -> str:
        """Returns default endpoint URL.
        RETURNS (str): Default endpoint URL.
        """

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
    def supported_models(self) -> Set[str]:
        """Set of supported models.
        RETURNS (Set[str]): Set of supported models.
        """

    @property
    @abc.abstractmethod
    def credentials(self) -> Dict[str, str]:
        """Get credentials for the LLM API.
        RETURNS (Dict[str, str]): Credentials.
        """

    def retry(
        self, call_api: Callable[[], requests.Response],
    ) -> requests.Response:
        """Retry a call to an API if we get a non-ok status code.
        This function automatically retries a request if it catches a response
        with an error code in `error_codes`. The amount of timeout also increases
        exponentially every time we retry.
        call_api (Callable[[], requests.Response]): Call to API.
        error_codes (Tuple[int]): Error codes indicating unsuccessful calls.
        RETURNS (requests.Response): Response of last call.
        """
        timeout = self._timeout
        response = call_api()
        i = 0

        # We don't want to retry on every non-ok status code. Some are about
        # incorrect inputs, etc. and we want to terminate on those.
        while i < self._max_tries and response.status_code in self._retry_error_codes:
            time.sleep(timeout)
            i += 1
            timeout = timeout * 2  # Increase timeout everytime you retry

        if response.status_code in self._retry_error_codes:
            raise ConnectionError(
                f"OpenAI API could not be reached within {self._timeout} seconds in {self._max_tries} attempts. Check "
                f"your network connection and the availability of the OpenAI API."
            )

        return response
