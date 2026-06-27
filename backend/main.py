import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import router as api_router
from backend.config.settings import get_settings
from backend.rag.embeddings import EmbeddingService
from backend.rag.llm import LLMService
from backend.rag.pipeline import RAGPipeline
from backend.rag.retrieval import RetrievalService
from backend.services.ingest_service import IngestService
from backend.vectordb.milvus_db import MilvusStore

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title=settings.app_name, version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=settings.api_v1_prefix)


@app.on_event("startup")
async def on_startup() -> None:
    settings.resolved_upload_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Starting %s", settings.app_name)

    embedding_service = EmbeddingService(settings)
    vector_store = MilvusStore(settings)
    retrieval_service = RetrievalService(settings, vector_store, embedding_service)
    llm_service = LLMService(settings)

    app.state.ingest_service = IngestService(settings, vector_store, embedding_service)
    app.state.rag_pipeline = RAGPipeline(settings, retrieval_service, llm_service)

    logger.info("Application startup complete")
