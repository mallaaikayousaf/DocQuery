"""
Configuration file for the project.
Loads environment variables and stores constants.
"""

import os

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):
        pass

# Load environment variables from .env file
load_dotenv()

# API Keys
API_KEY = os.getenv("API_KEY")

# Model Configurations
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.1-8b-instant"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 10  # Retrieve more chunks to give the LLM broader document coverage

# Paths
DATA_DIR = "data"
VECTOR_DB_PATH = "chroma_db"

# Validate API key
if not API_KEY:
    print("⚠️ WARNING: API_KEY not found in .env file")
    print("Please add it to use Groq LLM")