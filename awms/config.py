import os
from dataclasses import dataclass
from typing import Dict, Optional

from dotenv import load_dotenv

# Load environment variables
if not load_dotenv():
    raise EnvironmentError("Failed to load .env file")


@dataclass
class LLMConfig:
    model: str
    api_key: str
    endpoint: Optional[str] = None
    api_version: Optional[str] = None
    deployment_name: Optional[str] = None


# Constants
DEFAULT_OPENAI_MODEL = "gpt-4"
DEFAULT_AZURE_MODEL = "gpt-4"


def get_llm_config() -> Dict:
    api_source = os.getenv("API_SOURCE", "openai").lower()

    if api_source == "openai":
        api_key = os.getenv("OPENAPI_API_KEY")
        if not api_key:
            raise ValueError("OPENAPI_API_KEY not found in environment variables")

        return LLMConfig(model=f"openai/{os.getenv('MODEL_NAME', DEFAULT_OPENAI_MODEL)}", api_key=api_key).__dict__

    elif api_source == "azure":
        api_key = os.getenv("AZURE_OPENAPI_API_KEY")
        endpoint = os.getenv("AZURE_ENDPOINT_URL")
        api_version = os.getenv("AZURE_API_VERSION")

        if not all([api_key, endpoint, api_version]):
            raise ValueError("Missing required Azure configuration")

        os.environ.update({"AZURE_API_KEY": api_key, "AZURE_API_BASE": endpoint, "AZURE_API_VERSION": api_version})

        return LLMConfig(
            model=f"azure/{os.getenv('MODEL_NAME', DEFAULT_AZURE_MODEL)}",
            api_key=api_key,
            endpoint=endpoint,
            api_version=api_version,
            deployment_name=os.getenv("DEPLOYMENT_NAME"),
        ).__dict__

    raise ValueError(f"Unsupported API_SOURCE: {api_source}")


LLM_CONFIG = get_llm_config()
# remove None values
LLM_CONFIG = {k: v for k, v in LLM_CONFIG.items() if v is not None}

# LiteLLM specific settings
MAX_TOKENS = 1500
TEMPERATURE = 0.1
