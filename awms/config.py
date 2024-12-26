# Change this to your own API key
# Here we are using Azure OpenAI

import os

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

API_SOURCE = os.getenv("API_SOURCE", "openai")

# LLM Configuration

if API_SOURCE == "openai":
    OPENAPI_API_KEY = os.getenv("OPENAPI_API_KEY")
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4")
    LLM_CONFIG = {
        "model": f"openai/{MODEL_NAME}",
        "api_key": OPENAPI_API_KEY,
    }

elif API_SOURCE == "azure":
    OPENAPI_API_KEY = os.getenv("AZURE_OPENAPI_API_KEY")
    AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT_URL")
    AZURE_API_VERSION = os.getenv("AZURE_API_VERSION")
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
    DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")

    MODEL_TYPE = "azure"
    os.environ["AZURE_API_KEY"] = OPENAPI_API_KEY
    os.environ["AZURE_API_BASE"] = AZURE_ENDPOINT
    os.environ["AZURE_API_VERSION"] = AZURE_API_VERSION

    LLM_CONFIG = {
        "model": f"azure/{MODEL_NAME}",
        "api_key": OPENAPI_API_KEY,
        "api_base": AZURE_ENDPOINT,
        "api_version": AZURE_API_VERSION,
    }

# LiteLLM specific settings
MAX_TOKENS = 1500
TEMPERATURE = 0.1
