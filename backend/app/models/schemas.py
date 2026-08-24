from pydantic import BaseModel


class PageContent(BaseModel):
    page_number: int
    text: str
    blocks: list[dict]


class SectionBlock(BaseModel):
    section_id: str
    section_title: str
    page_start: int
    page_end: int
    text: str
    order_index: int


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    section_id: str
    section_title: str
    page_start: int
    page_end: int
    order_index: int
    text: str
    token_count: int


class ChromaQueryResult(BaseModel):
    ids: list[str]
    documents: list[str]
    metadatas: list[dict]
    distances: list[float]


class HybridSearchResult(BaseModel):
    chunk_id: str
    text: str
    section_title: str
    page_start: int
    page_end: int
    semantic_score: float
    lexical_score: float
    combined_score: float


class RerankedChunk(BaseModel):
    chunk_id: str
    text: str
    section_title: str
    page_start: int
    page_end: int
    relevance_score: float


class LLMAnswer(BaseModel):
    answer_text: str
    grounded: bool
    raw_finish_reason: str | None = None


class SourceCitation(BaseModel):
    section_title: str
    page_start: int
    page_end: int
    relevance_score: float
    excerpt: str


class DebugInfo(BaseModel):
    hybrid_results: list[HybridSearchResult]
    reranked_results: list[RerankedChunk]
    latency_ms: dict[str, float]


class QueryRequest(BaseModel):
    question: str
    document_id: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceCitation]
    document_id: str
    debug: DebugInfo | None = None


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    chunk_count: int
    page_count: int
    status: str


class DocumentSummary(BaseModel):
    document_id: str
    filename: str
    chunk_count: int
    page_count: int
    uploaded_at: str
