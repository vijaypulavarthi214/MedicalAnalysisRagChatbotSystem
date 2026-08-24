import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("GROQ_API_KEY", "test-groq-key")
os.environ.setdefault("GROQ_MODEL", "llama-3.3-70b-versatile")
os.environ.setdefault("COHERE_API_KEY", "test-cohere-key")
os.environ.setdefault("CHROMA_PERSIST_DIRECTORY", "./data/test_chroma")
