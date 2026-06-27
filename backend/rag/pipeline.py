from dataclasses import dataclass

from backend.config.settings import Settings
from backend.rag.llm import LLMService
from backend.rag.prompts import build_messages
from backend.rag.retrieval import RetrievedChunk, RetrievalService


@dataclass(slots=True)
class QAResult:
    answer: str
    sources: list[RetrievedChunk]


class RAGPipeline:
    def __init__(
        self,
        settings: Settings,
        retrieval_service: RetrievalService,
        llm_service: LLMService,
    ) -> None:
        self.settings = settings
        self.retrieval_service = retrieval_service
        self.llm_service = llm_service

    async def answer_question(self, question: str, top_k: int | None = None) -> QAResult:
        sources = self.retrieval_service.search(question, top_k=top_k)
        messages = build_messages(question, sources, max_chars=self.settings.max_context_chars)
        answer = await self.llm_service.generate_answer(messages)
        return QAResult(answer=answer, sources=sources)
