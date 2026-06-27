import logging
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool

from backend.config.settings import get_settings
from backend.models.schemas import HealthResponse, QueryRequest, QueryResponse, SourceChunkResponse, UploadResponse
from backend.rag.pipeline import RAGPipeline
from backend.services.ingest_service import IngestService

logger = logging.getLogger(__name__)
router = APIRouter()


def _get_state_service(request: Request, name: str):
    service = getattr(request.app.state, name, None)
    if service is None:
        raise HTTPException(status_code=503, detail=f"{name} is not ready")
    return service


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        app_name=settings.app_name,
        milvus_collection=settings.milvus_collection_name,
        llm_model=settings.llm_model,
    )


@router.post("/upload", response_model=UploadResponse)
async def upload_pdf(request: Request, file: UploadFile = File(...)) -> UploadResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="A file name is required")
    if Path(file.filename).suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    file_bytes = await file.read()
    settings = get_settings()
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds maximum size of {settings.max_upload_mb} MB",
        )

    ingest_service: IngestService = _get_state_service(request, "ingest_service")

    try:
        result = await run_in_threadpool(ingest_service.ingest_pdf, file_bytes, file.filename)
    except Exception as exc:
        logger.exception("Upload failed for %s", file.filename)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return UploadResponse(
        document_id=result.document_id,
        filename=result.filename,
        stored_path=result.stored_path,
        page_count=result.page_count,
        chunk_count=result.chunk_count,
        embedded_count=result.embedded_count,
    )


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: Request, payload: QueryRequest) -> QueryResponse:
    pipeline: RAGPipeline = _get_state_service(request, "rag_pipeline")

    try:
        result = await pipeline.answer_question(payload.question, top_k=payload.top_k)
    except Exception as exc:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return QueryResponse(
        question=payload.question,
        answer=result.answer,
        source_chunks=[
            SourceChunkResponse(
                document_id=chunk.document_id,
                source_filename=chunk.source_filename,
                page_number=chunk.page_number,
                chunk_index=chunk.chunk_index,
                score=chunk.score,
                chunk_text=chunk.chunk_text,
            )
            for chunk in result.sources
        ],
        retrieved_chunk_count=len(result.sources),
    )
