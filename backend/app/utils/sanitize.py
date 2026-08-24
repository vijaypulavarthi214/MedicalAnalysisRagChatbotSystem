import re
from pathlib import Path

_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]")


def sanitize_filename(raw_filename: str) -> str:
    name = Path(raw_filename or "document.pdf").name
    name = _UNSAFE_CHARS.sub("_", name)
    return name or "document.pdf"
