from types import SimpleNamespace

import groq
import httpx
import pytest

from app.errors import GenerationError
from app.llm.groq_client import NOT_FOUND_MESSAGE, generate_answer
from app.models.schemas import RerankedChunk


def _chunk():
    return RerankedChunk(
        chunk_id="a",
        text="Lisinopril 10mg daily",
        section_title="MEDICATIONS",
        page_start=2,
        page_end=2,
        relevance_score=0.9,
    )


def _response(status_code=401):
    return httpx.Response(
        status_code,
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
        json={"error": {"message": "boom"}},
    )


class FakeGroqClient:
    def __init__(self, result=None, error=None):
        self._result = result
        self._error = error
        self.last_kwargs = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        self.last_kwargs = kwargs
        if self._error is not None:
            raise self._error
        return self._result


def _chat_completion(text: str, finish_reason: str = "stop"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason=finish_reason)]
    )


@pytest.mark.asyncio
async def test_generate_answer_returns_grounded_answer():
    client = FakeGroqClient(result=_chat_completion("The patient takes Lisinopril 10mg daily."))

    result = await generate_answer(client, "llama-3.3-70b-versatile", "What medication?", [_chunk()])

    assert result.answer_text == "The patient takes Lisinopril 10mg daily."
    assert result.grounded is True
    assert result.raw_finish_reason == "stop"


@pytest.mark.asyncio
async def test_generate_answer_requests_hidden_reasoning():
    client = FakeGroqClient(result=_chat_completion("answer"))

    await generate_answer(client, "qwen/qwen3.6-27b", "question", [_chunk()])

    assert client.last_kwargs["reasoning_format"] == "hidden"


@pytest.mark.asyncio
async def test_generate_answer_raises_when_truncated_before_any_answer():
    client = FakeGroqClient(result=_chat_completion("", finish_reason="length"))

    with pytest.raises(GenerationError, match="ran out of output tokens"):
        await generate_answer(client, "qwen/qwen3.6-27b", "question", [_chunk()])


@pytest.mark.asyncio
async def test_generate_answer_marks_not_grounded_on_fixed_message():
    client = FakeGroqClient(result=_chat_completion(NOT_FOUND_MESSAGE))

    result = await generate_answer(client, "llama-3.3-70b-versatile", "What is the diagnosis?", [_chunk()])

    assert result.grounded is False
    assert result.answer_text == NOT_FOUND_MESSAGE


@pytest.mark.asyncio
async def test_generate_answer_raises_clear_message_on_bad_api_key():
    error = groq.AuthenticationError("bad key", response=_response(401), body=None)
    client = FakeGroqClient(error=error)

    with pytest.raises(GenerationError, match="rejected the API key"):
        await generate_answer(client, "llama-3.3-70b-versatile", "question", [_chunk()])


@pytest.mark.asyncio
async def test_generate_answer_raises_clear_message_on_unknown_model():
    error = groq.NotFoundError("not found", response=_response(404), body=None)
    client = FakeGroqClient(error=error)

    with pytest.raises(GenerationError, match="was not found"):
        await generate_answer(client, "bogus-model", "question", [_chunk()])


@pytest.mark.asyncio
async def test_generate_answer_raises_clear_message_on_rate_limit():
    error = groq.RateLimitError("rate limited", response=_response(429), body=None)
    client = FakeGroqClient(error=error)

    with pytest.raises(GenerationError, match="rate limit"):
        await generate_answer(client, "llama-3.3-70b-versatile", "question", [_chunk()])


@pytest.mark.asyncio
async def test_generate_answer_raises_clear_message_on_bad_request():
    error = groq.BadRequestError("invalid request", response=_response(400), body=None)
    client = FakeGroqClient(error=error)

    with pytest.raises(GenerationError, match="rejected the request"):
        await generate_answer(client, "llama-3.3-70b-versatile", "question", [_chunk()])


@pytest.mark.asyncio
async def test_generate_answer_raises_generation_error_on_server_error():
    error = groq.APIStatusError("server error", response=_response(500), body=None)
    client = FakeGroqClient(error=error)

    with pytest.raises(GenerationError):
        await generate_answer(client, "llama-3.3-70b-versatile", "question", [_chunk()])


@pytest.mark.asyncio
async def test_generate_answer_raises_generation_error_on_connection_failure():
    error = groq.APIConnectionError(
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    )
    client = FakeGroqClient(error=error)

    with pytest.raises(GenerationError):
        await generate_answer(client, "llama-3.3-70b-versatile", "question", [_chunk()])
