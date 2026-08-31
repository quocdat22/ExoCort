# Phase 2: Embedding & Vector Storage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Xây dựng Phase 2 thực hiện nhúng vector văn bản theo lô qua API `jina-embeddings-v5-omni-small` và lưu trữ vào Vector Store hỗ trợ lọc cô lập theo `workspace_id`.

**Architecture:** Nhận `List[Chunk]` từ Phase 1 -> Gom nhóm theo batch (32 chunks) -> Gọi Jina Embedding API tạo dense vector -> Lưu trữ vào `ScopedVectorStore` với chuẩn hóa vector L2 để tìm kiếm Cosine Similarity.

**Tech Stack:** Python 3.10+, `httpx`, `numpy`, `pytest`, `pytest-asyncio`.

**Spec:** `docs/superpowers/specs/2026-08-31-ebook-rag-baseline-hld.md`

## Global Constraints

- **Input Hand-off**: Nhận `List[Chunk]` đã chuẩn hóa từ Phase 1.
- **Embedding Model**: `jina-embeddings-v5-omni-small` qua Jina AI API (`https://api.jina.ai/v1/embeddings`).
- **Batch Size**: 32 chunks / request.
- **Isolation**: Vector Store phải lọc tuyệt đối theo `workspace_id`, không rò rỉ kết quả giữa các Workspace.
- **Output Hand-off**: Cung cấp `ScopedVectorStore` có phương thức `search_scoped()` cho Phase 3.

---

### Task 1: Scoped Vector Storage Engine with Cosine Similarity

**Files:**
- Create: `src/ebook_rag/storage/vector_store.py`
- Test: `tests/storage/test_vector_store.py`

**Interfaces:**
- Consumes: `Chunk` from `src.ebook_rag.core.models`
- Produces: `ScopedVectorStore.add_chunks()`, `ScopedVectorStore.search_scoped()`, `ScopedVectorStore.count()`

- [ ] **Step 1: Write failing unit test**

```python
# tests/storage/test_vector_store.py
import pytest
from src.ebook_rag.storage.vector_store import ScopedVectorStore
from src.ebook_rag.core.models import Chunk

def test_scoped_vector_store_isolation_and_similarity():
    store = ScopedVectorStore()

    chunk_ws1 = Chunk(
        chunk_id="c1", book_id="b1", book_title="Clean Code",
        workspace_id="ws_1", page_number=10, text_content="Functions should be small.",
        chunk_index=0
    )
    chunk_ws2 = Chunk(
        chunk_id="c2", book_id="b2", book_title="Cooking 101",
        workspace_id="ws_2", page_number=5, text_content="Preheat the oven.",
        chunk_index=0
    )

    vec1 = [1.0, 0.0, 0.0]
    vec2 = [0.0, 1.0, 0.0]

    store.add_chunks([chunk_ws1], [vec1])
    store.add_chunks([chunk_ws2], [vec2])

    # Search in ws_1: MUST return chunk from ws_1, NEVER chunk from ws_2
    query_vec = [0.95, 0.05, 0.0]
    results = store.search_scoped(workspace_id="ws_1", query_vector=query_vec, top_k=5)

    assert len(results) == 1
    assert results[0][0].chunk_id == "c1"
    assert results[0][0].workspace_id == "ws_1"
    assert results[0][1] > 0.9  # Cosine similarity score

def test_scoped_vector_store_empty_workspace():
    store = ScopedVectorStore()
    results = store.search_scoped(workspace_id="non_existent_ws", query_vector=[1.0, 0.0, 0.0], top_k=5)
    assert results == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/storage/test_vector_store.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/ebook_rag/storage/vector_store.py
from typing import List, Tuple, Dict
import numpy as np
from src.ebook_rag.core.models import Chunk

class ScopedVectorStore:
    def __init__(self):
        # workspace_id -> list of (Chunk, normalized_np_array)
        self._store: Dict[str, List[Tuple[Chunk, np.ndarray]]] = {}

    def add_chunks(self, chunks: List[Chunk], vectors: List[List[float]]) -> None:
        for chunk, vec in zip(chunks, vectors):
            ws_id = chunk.workspace_id
            if ws_id not in self._store:
                self._store[ws_id] = []
            
            norm_vec = np.array(vec, dtype=np.float32)
            norm = np.linalg.norm(norm_vec)
            if norm > 0:
                norm_vec = norm_vec / norm
            self._store[ws_id].append((chunk, norm_vec))

    def search_scoped(
        self,
        workspace_id: str,
        query_vector: List[float],
        top_k: int = 5
    ) -> List[Tuple[Chunk, float]]:
        if workspace_id not in self._store:
            return []

        q_vec = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        scored_chunks: List[Tuple[Chunk, float]] = []
        for chunk, doc_vec in self._store[workspace_id]:
            sim = float(np.dot(q_vec, doc_vec))
            scored_chunks.append((chunk, sim))

        # Sort descending by cosine similarity
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return scored_chunks[:top_k]

    def count(self, workspace_id: str) -> int:
        return len(self._store.get(workspace_id, []))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/storage/test_vector_store.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ebook_rag/storage/ tests/storage/
git commit -m "feat(phase-2): add scoped vector store engine"
```

---

### Task 2: Jina AI Embedding Client with Batch Processing

**Files:**
- Create: `src/ebook_rag/embedding/jina_client.py`
- Test: `tests/embedding/test_jina_client.py`

**Interfaces:**
- Consumes: `RAGConfig` from `src.ebook_rag.core.config`
- Produces: `JinaEmbeddingClient.embed_texts()`

- [ ] **Step 1: Write failing unit test**

```python
# tests/embedding/test_jina_client.py
import pytest
from unittest.mock import AsyncMock, patch
from src.ebook_rag.embedding.jina_client import JinaEmbeddingClient
from src.ebook_rag.core.config import RAGConfig

@pytest.mark.asyncio
async def test_jina_client_batching():
    config = RAGConfig(
        embedding_model="jina-embeddings-v5-omni-small",
        embedding_batch_size=2,
        jina_api_key="mock_key"
    )
    client = JinaEmbeddingClient(config)

    mock_resp_payload_1 = {
        "data": [{"embedding": [0.1, 0.2]}, {"embedding": [0.3, 0.4]}]
    }
    mock_resp_payload_2 = {
        "data": [{"embedding": [0.5, 0.6]}]
    }

    texts = ["text1", "text2", "text3"]  # 3 items -> 2 batches (2 + 1)

    with patch("httpx.AsyncClient.post") as mock_post:
        mock_post.side_effect = [
            AsyncMock(json=lambda: mock_resp_payload_1, raise_for_status=lambda: None),
            AsyncMock(json=lambda: mock_resp_payload_2, raise_for_status=lambda: None)
        ]

        embeddings = await client.embed_texts(texts)
        assert len(embeddings) == 3
        assert embeddings[0] == [0.1, 0.2]
        assert embeddings[2] == [0.5, 0.6]
        assert mock_post.call_count == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/embedding/test_jina_client.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/ebook_rag/embedding/jina_client.py
from typing import List
import httpx
from src.ebook_rag.core.config import RAGConfig

class JinaEmbeddingClient:
    def __init__(self, config: RAGConfig):
        self.model = config.embedding_model
        self.api_key = config.jina_api_key
        self.batch_size = config.embedding_batch_size
        self.api_url = "https://api.jina.ai/v1/embeddings"

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        all_embeddings: List[List[float]] = []
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                payload = {
                    "model": self.model,
                    "input": batch
                }
                resp = await client.post(self.api_url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                for item in data.get("data", []):
                    all_embeddings.append(item["embedding"])

        return all_embeddings
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/embedding/test_jina_client.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ebook_rag/embedding/ tests/embedding/
git commit -m "feat(phase-2): add Jina embedding client with batching"
```
