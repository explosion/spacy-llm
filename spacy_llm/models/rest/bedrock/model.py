import json
import os
import warnings
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..base import REST


class Models(str, Enum):
    # Completion models
    TITAN_EXPRESS = "amazon.titan-text-express-v1"
    TITAN_LITE = "amazon.titan-text-lite-v1"
    AI21_JURASSIC_ULTRA = "ai21.j2-ultra-v1"
    AI21_JURASSIC_MID = "ai21.j2-mid-v1"


TITAN_PARAMS = ["maxTokenCount", "stopSequences", "temperature", "topP"]
AI21_JURASSIC_PARAMS = [
    "maxTokens",
    "temperature",
    "topP",
    "countPenalty",
    "presencePenalty",
    "frequencyPenalty",
]


class Bedrock(REST):
    def __init__(
        self,
        model_id: str,
        region: str,
        config: Dict[Any, Any],
        max_tries: int = 5,
    ):
        self._region = region
        self._model_id = model_id
        self._max_tries = max_tries
        self.strict = True
        self.endpoint = f"https://bedrock-runtime.{self._region}.amazonaws.com"
        self._config = {}

        if self._model_id in [Models.TITAN_EXPRESS, Models.TITAN_LITE]:
            config_params = TITAN_PARAMS
        if self._model_id in [Models.AI21_JURASSIC_ULTRA, Models.AI21_JURASSIC_MID]:
            config_params = AI21_JURASSIC_PARAMS

        for i in config_params:
            self._config[i] = config[i]

        super().__init__(
            name=model_id,
            config=self._config,
            max_tries=max_tries,
            strict=True,
            endpoint="",
            interval=3,
            max_request_time=30,
        )

    def get_session_kwargs(self) -> Dict[str, Optional[str]]:

        # Fetch and check the credentials
        profile = os.getenv("AWS_PROFILE") if not None else "default"
        secret_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        session_token = os.getenv("AWS_SESSION_TOKEN")

        if profile is None:
            warnings.warn(
                "Could not find the AWS_PROFILE to access the Amazon Bedrock . Ensure you have an AWS_PROFILE "
                "set up by making it available as an environment variable AWS_PROFILE."
            )

        if secret_key_id is None:
            warnings.warn(
                "Could not find the AWS_ACCESS_KEY_ID to access the Amazon Bedrock . Ensure you have an AWS_ACCESS_KEY_ID "
                "set up by making it available as an environment variable AWS_ACCESS_KEY_ID."
            )

        if secret_access_key is None:
            warnings.warn(
                "Could not find the AWS_SECRET_ACCESS_KEY to access the Amazon Bedrock . Ensure you have an AWS_SECRET_ACCESS_KEY "
                "set up by making it available as an environment variable AWS_SECRET_ACCESS_KEY."
            )

        if session_token is None:
            warnings.warn(
                "Could not find the AWS_SESSION_TOKEN to access the Amazon Bedrock . Ensure you have an AWS_SESSION_TOKEN "
                "set up by making it available as an environment variable AWS_SESSION_TOKEN."
            )

        assert secret_key_id is not None
        assert secret_access_key is not None
        assert session_token is not None

        session_kwargs = {
            "profile_name": profile,
            "region_name": self._region,
            "aws_access_key_id": secret_key_id,
            "aws_secret_access_key": secret_access_key,
            "aws_session_token": session_token,
        }
        return session_kwargs

    def __call__(self, prompts: Iterable[str]) -> Iterable[str]:
        api_responses: List[str] = []
        prompts = list(prompts)

        def _request(json_data: str) -> str:
            try:
                import boto3
            except ImportError as err:
                warnings.warn(
                    "To use Bedrock, you need to install boto3. Use pip install boto3 "
                )
                raise err
            from botocore.config import Config

            session_kwargs = self.get_session_kwargs()
            session = boto3.Session(**session_kwargs)
            api_config = Config(retries=dict(max_attempts=self._max_tries))
            bedrock = session.client(service_name="bedrock-runtime", config=api_config)
            accept = "application/json"
            contentType = "application/json"
            r = bedrock.invoke_model(
                body=json_data,
                modelId=self._model_id,
                accept=accept,
                contentType=contentType,
            )
            if self._model_id in [Models.TITAN_EXPRESS, Models.TITAN_LITE]:
                responses = json.loads(r["body"].read().decode())["results"][0][
                    "outputText"
                ]
            elif self._model_id in [
                Models.AI21_JURASSIC_ULTRA,
                Models.AI21_JURASSIC_MID,
            ]:
                responses = json.loads(r["body"].read().decode())["completions"][0][
                    "data"
                ]["text"]

            return responses

        for prompt in prompts:
            if self._model_id in [Models.TITAN_EXPRESS, Models.TITAN_LITE]:
                responses = _request(
                    json.dumps(
                        {"inputText": prompt, "textGenerationConfig": self._config}
                    )
                )
            if self._model_id in [Models.AI21_JURASSIC_ULTRA, Models.AI21_JURASSIC_MID]:
                responses = _request(json.dumps({"prompt": prompt, **self._config}))

            api_responses.append(responses)

        return api_responses

    def _verify_auth(self) -> None:
        try:
            import boto3
            from botocore.exceptions import NoCredentialsError

            session_kwargs = self.get_session_kwargs()
            session = boto3.Session(**session_kwargs)
            bedrock = session.client(service_name="bedrock")
            bedrock.list_foundation_models()
        except NoCredentialsError:
            raise NoCredentialsError

    @property
    def credentials(self) -> Dict[str, Optional[str]]:  # type: ignore
        return self.get_session_kwargs()

    @classmethod
    def get_model_names(self) -> Tuple[str, ...]:
        return (
            "amazon.titan-text-express-v1",
            "amazon.titan-text-lite-v1",
            "ai21.j2-ultra-v1",
            "ai21.j2-mid-v1",
        )
