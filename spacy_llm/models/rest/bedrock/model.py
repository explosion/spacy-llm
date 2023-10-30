import json
import os
import warnings
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional


class Models(str, Enum):
    # Completion models
    TITAN_EXPRESS = "amazon.titan-text-express-v1"
    TITAN_LITE = "amazon.titan-text-lite-v1"


class Bedrock:
    def __init__(
        self, model_id: str, region: str, config: Dict[Any, Any], max_retries: int = 5
    ):
        self._region = region
        self._model_id = model_id
        self._config = config
        self._max_retries = max_retries

    def get_session_kwargs(self) -> Dict[str, Optional[str]]:

        # Fetch and check the credentials
        profile = os.getenv("AWS_PROFILE") if not None else ""
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
            api_config = Config(retries=dict(max_attempts=self._max_retries))
            bedrock = session.client(service_name="bedrock-runtime", config=api_config)
            accept = "application/json"
            contentType = "application/json"
            r = bedrock.invoke_model(
                body=json_data,
                modelId=self._model_id,
                accept=accept,
                contentType=contentType,
            )
            responses = json.loads(r["body"].read().decode())["results"][0][
                "outputText"
            ]
            return responses

        for prompt in prompts:
            if self._model_id in [Models.TITAN_LITE, Models.TITAN_EXPRESS]:
                responses = _request(
                    json.dumps(
                        {"inputText": prompt, "textGenerationConfig": self._config}
                    )
                )

            api_responses.append(responses)

        return api_responses
