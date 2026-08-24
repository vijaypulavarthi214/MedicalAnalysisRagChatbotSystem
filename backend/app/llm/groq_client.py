import groq

from app.errors import GenerationError
from app.models.schemas import LLMAnswer, RerankedChunk

NOT_FOUND_MESSAGE = "This information was not found in the provided document."

_SYSTEM_PROMPT = (
    "You are a medical document assistant. Answer the user's question ONLY "
    "using the numbered context excerpts below, taken from the patient's "
    "uploaded document. Do not use outside knowledge. Do not diagnose, "
    "recommend treatment, or infer anything beyond what is explicitly "
    "stated in the context. If the answer is not present in the context, "
    f'respond with exactly this sentence and nothing else: "{NOT_FOUND_MESSAGE}"'
)


def _build_user_message(question: str, context_chunks: list[RerankedChunk]) -> str:
    context_block = "\n\n".join(
        f"[{i + 1}] (Section: {c.section_title}, Page: {c.page_start}-{c.page_end})\n{c.text}"
        for i, c in enumerate(context_chunks)
    )
    return f"Context:\n{context_block}\n\nQuestion: {question}"


async def generate_answer(
    client: groq.AsyncGroq,
    model: str,
    question: str,
    chunks: list[RerankedChunk],
    max_tokens: int = 2000,
) -> LLMAnswer:
    try:
        response = await client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            # Reasoning models (qwen3, deepseek-r1, etc.) emit their chain-of-thought
            # as part of the completion by default — "hidden" strips it so `content`
            # is only the final answer. Non-reasoning models ignore this param.
            reasoning_format="hidden",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_message(question, chunks)},
            ],
        )
    except groq.AuthenticationError as exc:
        raise GenerationError(
            "Groq rejected the API key (401). Check GROQ_API_KEY."
        ) from exc
    except groq.NotFoundError as exc:
        raise GenerationError(
            f"Groq model '{model}' was not found (404). Check GROQ_MODEL."
        ) from exc
    except groq.RateLimitError as exc:
        retry_after = exc.response.headers.get("retry-after", "a short while")
        raise GenerationError(
            f"Groq rate limit reached. Wait {retry_after} and try again."
        ) from exc
    except groq.BadRequestError as exc:
        raise GenerationError(f"Groq rejected the request: {exc.message}") from exc
    except groq.PermissionDeniedError as exc:
        raise GenerationError(f"Groq denied the request: {exc.message}") from exc
    except groq.APIStatusError as exc:
        raise GenerationError(f"Groq returned HTTP {exc.status_code}: {exc.message}") from exc
    except groq.APIConnectionError as exc:
        raise GenerationError(f"Groq request failed: {type(exc).__name__}") from exc

    choice = response.choices[0]
    answer_text = (choice.message.content or "").strip()

    if not answer_text and choice.finish_reason == "length":
        raise GenerationError(
            "Groq ran out of output tokens before writing an answer (the model "
            "spent them on hidden reasoning). Increase max_tokens in generate_answer()."
        )

    grounded = answer_text != NOT_FOUND_MESSAGE
    return LLMAnswer(answer_text=answer_text, grounded=grounded, raw_finish_reason=choice.finish_reason)
