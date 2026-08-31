# Phase 2: Embedding & Vector Storage Implementation Plan (Managed with uv)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Xây dựng Phase 2 thực hiện nhúng vector văn bản theo lô qua API `jina-embeddings-v5-omni-small` và lưu trữ vào Vector Store hỗ trợ lọc cô lập theo `workspace_id` sử dụng ChromaDB. Quản lý môi trường và thực thi với `uv`.

**Architecture:** Nhận `List[Chunk]` từ Phase 1 -> Gom nhóm theo batch (32 chunks) -> Gọi Jina Embedding API tạo dense vector -> Lưu trữ vào ChromaDB (thông qua `ScopedVectorStore`) để persistence và tìm kiếm Cosine Similarity.

**Tech Stack:** `uv`, Python 3.10+, `httpx`, `numpy`, `pytest`, `pytest-asyncio`, `chromadb`, `tenacity`.

**Spec:** `docs/superpowers/specs/2026-08-31-exocort-baseline-hld.md`

## Global Constraints

- **Environment & Execution**: Toàn bộ các lệnh chạy test, cài đặt dependencies và thực thi phải thông qua **`uv`** (ví dụ: `uv run pytest`).
  ```bash
  uv add httpx numpy chromadb tenacity
  ```
- **Input Hand-off**: Nhận `List[Chunk]` đã chuẩn hóa từ Phase 1.
- **Embedding Model**: `jina-embeddings-v5-omni-small` qua Jina AI API (`https://api.jina.ai/v1/embeddings`).
- **Batch Size**: 32 chunks / request.
- **Isolation**: Vector Store phải lọc tuyệt đối theo `workspace_id`, không rò rỉ kết quả giữa các Workspace. Sử dụng ChromaDB làm persistent vector store.
- **Output Hand-off**: Cung cấp `ScopedVectorStore` có phương thức `search_scoped()` cho Phase 3.

---

### Task 1: Scoped Vector Storage Engine with ChromaDB

**Files:**
- Create: `src/exocort/storage/vector_store.py`
- Test: `tests/storage/test_vector_store.py`

**Interfaces:**
- Consumes: `Chunk` from `exocort.core.models`
- Produces: `ScopedVectorStore.add_chunks()`, `ScopedVectorStore.search_scoped()`, `ScopedVectorStore.count()`, `ScopedVectorStore.delete_by_book()`

- [ ] **Step 1: Write failing unit test**

```python
# tests/storage/test_vector_store.py
import pytest
from exocort.storage.vector_store import ScopedVectorStore
from exocort.core.models import Chunk


def test_scoped_vector_store_isolation(tmp_path):
    store = ScopedVectorStore(persist_dir=str(tmp_path / "chroma"))

    chunk_ws1 = Chunk(
        chunk_id="c1", book_id="b1", book_title="Clean Code",
        workspace_id="ws_1", page_start=10, page_end=10,
        text_content="Functions should be small.", chunk_index=0
    )
    chunk_ws2 = Chunk(
        chunk_id="c2", book_id="b2", book_title="Cooking 101",
        workspace_id="ws_2", page_start=5, page_end=5,
        text_content="Preheat the oven.", chunk_index=0
    )

    vec1 = [1.0] + [0.0] * 1023  # 1024-dim matching Jina v5 omni small
    vec2 = [0.0] + [1.0] + [0.0] * 1022

    store.add_chunks([chunk_ws1], [vec1])
    store.add_chunks([chunk_ws2], [vec2])

    # Search in ws_1: MUST return only chunk from ws_1
    query_vec = [0.95] + [0.05] + [0.0] * 1022
    results = store.search_scoped(workspace_id="ws_1", query_vector=query_vec, top_k=5)

    assert len(results) == 1
    assert results[0][0].chunk_id == "c1"
    assert results[0][0].workspace_id == "ws_1"


def test_scoped_vector_store_empty_workspace(tmp_path):
    store = ScopedVectorStore(persist_dir=str(tmp_path / "chroma"))
    results = store.search_scoped(
        workspace_id="non_existent",
        query_vector=[1.0] + [0.0] * 1023,
        top_k=5
    )
    assert results == []


def test_persistence_survives_reload(tmp_path):
    chroma_dir = str(tmp_path / "chroma")
    store = ScopedVectorStore(persist_dir=chroma_dir)
    chunk = Chunk(
        chunk_id="c1", book_id="b1", book_title="Book",
        workspace_id="ws_1", page_start=1, page_end=1,
        text_content="Test text.", chunk_index=0
    )
    store.add_chunks([chunk], [[1.0] + [0.0] * 1023])
    assert store.count() == 1

    # Reload from disk
    store2 = ScopedVectorStore(persist_dir=chroma_dir)
    assert store2.count() == 1
```

- [ ] **Step 2: Run test using uv to verify it fails**

Run: `uv run pytest tests/storage/test_vector_store.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/exocort/storage/vector_store.py
from typing import List, Tuple, Optional
from pathlib import Path
import chromadb
from chromadb.config import Settings
from exocort.core.models import Chunk


class ScopedVectorStore:
    """ChromaDB-backed vector store with workspace-scoped retrieval."""

    def __init__(self, persist_dir: str | Path = "./chroma_data",
                 collection_name: str = "ebook_chunks"):
        self.client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: List[Chunk], vectors: List[List[float]]) -> None:
        """Upsert chunks with their embeddings into ChromaDB."""
        if not chunks:
            return
        self.collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=vectors,
            documents=[c.text_content for c in chunks],
            metadatas=[
                {
                    "workspace_id": c.workspace_id,
                    "book_id": c.book_id,
                    "book_title": c.book_title,
                    "page_start": c.page_start,
                    "page_end": c.page_end,
                    "chunk_index": c.chunk_index,
                }
                for c in chunks
            ],
        )

    def search_scoped(
        self,
        workspace_id: str,
        query_vector: List[float],
        top_k: int = 5,
    ) -> List[Tuple[Chunk, float]]:
        """Retrieve top-k similar chunks filtered by workspace_id."""
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where={"workspace_id": workspace_id},
            include=["documents", "metadatas", "distances"],
        )

        if not results["ids"] or not results["ids"][0]:
            return []

        scored: List[Tuple[Chunk, float]] = []
        for i, chunk_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i]
            doc_text = results["documents"][0][i]
            distance = results["distances"][0][i]
            similarity = 1.0 - distance  # ChromaDB cosine distance → similarity

            chunk = Chunk(
                chunk_id=chunk_id,
                book_id=meta["book_id"],
                book_title=meta["book_title"],
                workspace_id=meta["workspace_id"],
                page_start=meta["page_start"],
                page_end=meta["page_end"],
                text_content=doc_text,
                chunk_index=meta["chunk_index"],
            )
            scored.append((chunk, round(similarity, 4)))

        return scored

    def delete_by_book(self, book_id: str) -> None:
        """Delete all chunks belonging to a book."""
        self.collection.delete(where={"book_id": book_id})

    def count(self) -> int:
        return self.collection.count()
```

- [ ] **Step 4: Run test using uv to verify it passes**

Run: `uv run pytest tests/storage/test_vector_store.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/exocort/storage/ tests/storage/
git commit -m "feat(phase-2): add scoped vector store engine with chromadb"
```

---

### Task 2: Jina AI Embedding Client with Dual Resilience Profile & Throttling

**Files:**
- Create: `src/exocort/embedding/jina_client.py`
- Test: `tests/embedding/test_jina_client.py`

**Interfaces:**
- Consumes: `RAGConfig` from `exocort.core.config`
- Produces: `JinaEmbeddingClient.embed_texts()`, `JinaEmbeddingClient.embed_query()`

- [ ] **Step 1: Write failing unit test**

```python
# tests/embedding/test_jina_client.py
import pytest
from unittest.mock import AsyncMock, patch
from pydantic import SecretStr
from exocort.embedding.jina_client import JinaEmbeddingClient
from exocort.core.config import RAGConfig

@pytest.mark.asyncio
async def test_jina_client_batching_and_throttling():
    config = RAGConfig(
        embedding_model="jina-embeddings-v5-omni-small",
        embedding_batch_size=2,
        jina_api_key=SecretStr("mock_key"),
        inter_batch_delay_sec=0.0
    )
    client = JinaEmbeddingClient(config)

    mock_resp_payload_1 = {
        "data": [{"embedding": [0.1] * 1024}, {"embedding": [0.3] * 1024}]
    }
    mock_resp_payload_2 = {
        "data": [{"embedding": [0.5] * 1024}]
    }

    texts = ["text1", "text2", "text3"]

    with patch("httpx.AsyncClient.post") as mock_post:
        mock_post.side_effect = [
            AsyncMock(json=lambda: mock_resp_payload_1, raise_for_status=lambda: None),
            AsyncMock(json=lambda: mock_resp_payload_2, raise_for_status=lambda: None)
        ]

        embeddings = await client.embed_texts(texts)
        assert len(embeddings) == 3
        assert embeddings[0] == [0.1] * 1024
        assert embeddings[2] == [0.5] * 1024
        assert mock_post.call_count == 2

@pytest.mark.asyncio
async def test_jina_client_embed_query_fast_profile():
    config = RAGConfig(
        jina_api_key=SecretStr("mock_key"),
        query_embedding_timeout=2.0
    )
    client = JinaEmbeddingClient(config)

    mock_resp_payload = {
        "data": [{"embedding": [0.9] * 1024}]
    }

    with patch("httpx.AsyncClient.post") as mock_post:
        mock_post.return_value = AsyncMock(
            json=lambda: mock_resp_payload,
            raise_for_status=lambda: None
        )
        vec = await client.embed_query("What is inheritance?")
        assert len(vec) == 1024
        assert vec[0] == 0.9
```

- [ ] **Step 2: Run test using uv to verify it fails**

Run: `uv run pytest tests/embedding/test_jina_client.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/exocort/embedding/jina_client.py
import asyncio
from typing import List
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, wait_fixed
from exocort.core.config import RAGConfig

class JinaEmbeddingClient:
    """Client for Jina Embedding API with Dual-Profile Resilience (Patient Ingest vs Agile Query)."""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.model = config.embedding_model
        self.api_key = (
            config.jina_api_key.get_secret_value()
            if hasattr(config.jina_api_key, "get_secret_value")
            else str(config.jina_api_key)
        )
        self.batch_size = config.embedding_batch_size
        self.api_url = "https://api.jina.ai/v1/embeddings"
        self.semaphore = asyncio.Semaphore(config.max_concurrent_embedding_batches)

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=30))
    async def _embed_batch_patient(self, texts: List[str]) -> List[List[float]]:
        """Patient retry profile for offline batch ingestion."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        async with self.semaphore:
            async with httpx.AsyncClient(timeout=self.config.ingestion_embedding_timeout) as client:
                payload = {"model": self.model, "input": texts}
                resp = await client.post(self.api_url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                return [item["embedding"] for item in data.get("data", [])]

    @retry(stop=stop_after_attempt(2), wait=wait_fixed(0.5))
    async def embed_query(self, query: str) -> List[float]:
        """Fast agile profile for interactive query-time search."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        async with httpx.AsyncClient(timeout=self.config.query_embedding_timeout) as client:
            payload = {"model": self.model, "input": [query]}
            resp = await client.post(self.api_url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            embeddings = [item["embedding"] for item in data.get("data", [])]
            return embeddings[0] if embeddings else [0.0] * self.config.embedding_dimension

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = await self._embed_batch_patient(batch)
            all_embeddings.extend(batch_embeddings)
            if self.config.inter_batch_delay_sec > 0:
                await asyncio.sleep(self.config.inter_batch_delay_sec)

        return all_embeddings
```

- [ ] **Step 4: Run test using uv to verify it passes**

Run: `uv run pytest tests/embedding/test_jina_client.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/exocort/embedding/ tests/embedding/
git commit -m "feat(phase-2): add Jina embedding client with dual resilience profiles and throttling"
```
