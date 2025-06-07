import os
from dotenv import load_dotenv

# Load environment variables from .env file at project root
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../../.env'))

# Optionally, expose config values as variables
HF_HOME = os.getenv("HF_HOME")
HF_HUB_CACHE = os.getenv("HF_HUB_CACHE")