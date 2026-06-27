from dataclasses import dataclass
import logging
import os
from typing import Any, Optional

import numpy as np

from backend.config.settings import Settings, get_settings
from backend.rag.chunking import ChunkRecord

logger = logging.getLogger(__name__)


# pymilvus validates os.environ["MILVUS_URI"] at import time (http(s) only).
# Never set MILVUS_URI to a .db path before "from pymilvus import ...".
# The real .db path is passed to MilvusClient(...) only when the store is created.
_IMPORT_PLACEHOLDER_URI = "http://127.0.0.1:19530"


def _bootstrap_pymilvus_environment(settings: Settings) -> str:
    """
    Load Milvus Lite and satisfy pymilvus import-time URI checks.

    For .db URIs: import milvus_lite, set a temporary http MILVUS_URI for import only,
    then pass the resolved .db path to MilvusClient(db_path) in _create_client().
    """
    uri = settings.resolved_milvus_uri
    if settings.uses_milvus_lite:
        try:
            import milvus_lite  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Milvus Lite is required when MILVUS_URI ends with .db. Install with: "
                'pip install "pymilvus[milvus-lite]>=3.0.0" milvus-lite'
            ) from exc
        os.environ["MILVUS_URI"] = _IMPORT_PLACEHOLDER_URI
        logger.debug("Milvus Lite client URI: %s", uri)
    else:
        os.environ["MILVUS_URI"] = uri
    return uri


_bootstrap_pymilvus_environment(get_settings())

from pymilvus import DataType, MilvusClient, MilvusException 


@dataclass(slots=True)
class VectorSearchHit:
    document_id: str
    source_filename: str
    page_number: int
    chunk_index: int
    chunk_text: str
    score: float


class MilvusStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.collection_name = settings.milvus_collection_name
        self._uri = settings.resolved_milvus_uri
        self._client = self._create_client()
        self._ensure_collection()

    def _effective_index_type(self) -> str:
        # Milvus Lite only supports FLAT (README); remote Milvus can use HNSW etc.
        if self.settings.uses_milvus_lite:
            return "FLAT"
        return self.settings.milvus_index_type.upper()

    def _create_client(self) -> MilvusClient:
        token: Optional[str] = getattr(self.settings, "milvus_token", None)
        logger.info(
            "Connecting to Milvus at %s (lite=%s)",
            self._uri,
            self.settings.uses_milvus_lite,
        )
        try:
            # Milvus Lite: local file path (official API: MilvusClient("./file.db"))
            if self.settings.uses_milvus_lite:
                return MilvusClient(self._uri)

            # Remote Milvus server
            if token:
                return MilvusClient(uri=self._uri, token=token)
            return MilvusClient(uri=self._uri)
        except MilvusException as exc:
            hint = ""
            if "19530" in str(exc):
                hint = (
                    " No Milvus server is running on localhost:19530. "
                    "For embedded storage, set MILVUS_URI to an absolute path ending in .db "
                    "(e.g. MILVUS_URI=./milvus_local.db) and install milvus-lite."
                )
            raise RuntimeError(f"Failed to connect to Milvus at {self._uri}.{hint}") from exc

    def _build_index_params(self):
        index_params = self._client.prepare_index_params()
        index_type = self._effective_index_type()
        kwargs: dict[str, Any] = {
            "field_name": "embedding",
            "index_type": index_type,
            "metric_type": self.settings.milvus_metric_type,
        }
        if index_type == "HNSW":
            kwargs["params"] = {
                "M": self.settings.milvus_m,
                "efConstruction": self.settings.milvus_ef_construction,
            }
        index_params.add_index(**kwargs)
        return index_params

    def _search_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {"metric_type": self.settings.milvus_metric_type}
        if self._effective_index_type() == "HNSW":
            params["params"] = {"ef": self.settings.milvus_ef_search}
        return params

    def _ensure_collection(self) -> None:
        try:
            if self._client.has_collection(self.collection_name):
                try:
                    self._client.load_collection(self.collection_name)
                except MilvusException:
                    logger.exception("Failed to load existing collection %s", self.collection_name)
                return

            index_type = self._effective_index_type()
            logger.info(
                "Creating Milvus collection %s (index=%s, lite=%s)",
                self.collection_name,
                index_type,
                self.settings.uses_milvus_lite,
            )

            schema = self._client.create_schema(auto_id=True, enable_dynamic_field=False)
            schema.add_field("id", DataType.INT64, is_primary=True)
            schema.add_field("document_id", DataType.VARCHAR, max_length=64)
            schema.add_field("source_filename", DataType.VARCHAR, max_length=512)
            schema.add_field("page_number", DataType.INT64)
            schema.add_field("chunk_index", DataType.INT64)
            schema.add_field("chunk_text", DataType.VARCHAR, max_length=65535)
            schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=self.settings.milvus_dimension)

            self._client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                index_params=self._build_index_params(),
            )

            try:
                self._client.create_index(collection_name=self.collection_name, field_name="embedding")
            except Exception:
                logger.debug("create_index not required or failed; continuing")

            self._client.load_collection(self.collection_name)

        except MilvusException as exc:
            logger.exception("Milvus operation failed: %s", exc)
            raise

    def insert_chunks(self, chunks: list[ChunkRecord], embeddings: np.ndarray) -> int:
        if not chunks:
            return 0
        if len(chunks) != len(embeddings):
            raise ValueError("Chunk count and embedding count must match")

        data = []
        for chunk, emb in zip(chunks, embeddings):
            data.append(
                {
                    "document_id": chunk.document_id,
                    "source_filename": chunk.source_filename,
                    "page_number": int(chunk.page_number),
                    "chunk_index": int(chunk.chunk_index),
                    "chunk_text": chunk.text,
                    "embedding": emb.tolist(),
                }
            )

        self._client.insert(collection_name=self.collection_name, data=data)
        try:
            self._client.flush([self.collection_name])
        except Exception:
            logger.debug("Flush not required or failed")

        logger.info("Inserted %s vectors into %s", len(data), self.collection_name)
        return len(data)

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[VectorSearchHit]:
        if query_embedding.ndim == 1:
            query_embedding = np.expand_dims(query_embedding, axis=0)

        results = self._client.search(
            collection_name=self.collection_name,
            data=query_embedding.tolist(),
            limit=top_k,
            search_params=self._search_params(),
            output_fields=["document_id", "source_filename", "page_number", "chunk_index", "chunk_text"],
        )

        hits: list[VectorSearchHit] = []
        for batch in results:
            for hit in batch:
                entity = getattr(hit, "entity", None) or (hit.get("entity") if isinstance(hit, dict) else {})
                score = (
                    getattr(hit, "distance", None)
                    or getattr(hit, "score", None)
                    or (hit.get("distance") if isinstance(hit, dict) else None)
                    or (hit.get("score") if isinstance(hit, dict) else None)
                )

                hits.append(
                    VectorSearchHit(
                        document_id=str(entity.get("document_id")),
                        source_filename=str(entity.get("source_filename")),
                        page_number=int(entity.get("page_number") or 0),
                        chunk_index=int(entity.get("chunk_index") or 0),
                        chunk_text=str(entity.get("chunk_text") or ""),
                        score=float(score) if score is not None else 0.0,
                    )
                )

        return hits
