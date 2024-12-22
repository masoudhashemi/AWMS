# Change this to your own API key
# Here we are using Azure OpenAI

import os

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# LLM Configuration
OPENAPI_API_KEY = os.getenv("OPENAPI_API_KEY")
AZURE_ENDPOINT = os.getenv("ENDPOINT_URL")
AZURE_API_VERSION = os.getenv("API_VERSION")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")

# LiteLLM specific settings
MAX_TOKENS = 1500
TEMPERATURE = 0.1

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
