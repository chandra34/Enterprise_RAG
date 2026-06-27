from dataclasses import dataclass

from backend.config.settings import Settings
from backend.rag.embeddings import EmbeddingService
from backend.vectordb.milvus_db import MilvusStore, VectorSearchHit


@dataclass(slots=True)
class RetrievedChunk:
    document_id: str
    source_filename: str
    page_number: int
    chunk_index: int
    chunk_text: str
    score: float


class RetrievalService:
    def __init__(
        self,
        settings: Settings,
        vector_store: MilvusStore,
        embedding_service: EmbeddingService,
    ) -> None:
        self.settings = settings
        self.vector_store = vector_store
        self.embedding_service = embedding_service

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        top_k = top_k or self.settings.top_k
        query_embedding = self.embedding_service.embed_query(query)
        hits = self.vector_store.search(query_embedding, top_k=top_k)
        return [_hit_to_chunk(hit) for hit in hits]


def _hit_to_chunk(hit: VectorSearchHit) -> RetrievedChunk:
    return RetrievedChunk(
        document_id=hit.document_id,
        source_filename=hit.source_filename,
        page_number=hit.page_number,
        chunk_index=hit.chunk_index,
        chunk_text=hit.chunk_text,
        score=hit.score,
    )
