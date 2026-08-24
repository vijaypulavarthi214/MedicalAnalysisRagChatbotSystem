class PipelineError(Exception):
    """Base class for all RAG pipeline failures. Never caught silently —
    every subclass maps to an explicit error response, never a fabricated
    or partial answer."""


class IngestionError(PipelineError):
    pass


class RetrievalError(PipelineError):
    pass


class RerankError(PipelineError):
    pass


class GenerationError(PipelineError):
    pass


class ConfigError(PipelineError):
    pass
