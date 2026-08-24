"""Structured logging. Callers must only pass PHI-free fields: document/chunk
UUIDs, counts, durations, section names, HTTP status codes, exception class
names. Never log chunk text, question text, filenames as provided by the
user, or LLM prompt/response content.
"""

import logging
import sys

_CONFIGURED = False


def configure_logging(level: int = logging.INFO) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s level=%(levelname)s logger=%(name)s msg=%(message)s"
    )
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
