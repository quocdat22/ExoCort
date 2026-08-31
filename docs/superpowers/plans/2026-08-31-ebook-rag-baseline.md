# E-Book RAG Engine (Baseline MVP) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Xây dựng hệ thống E-Book RAG Baseline MVP theo kiến trúc hộp đen với 4 Phase khép kín: Ingestion & Validation, Jina Embedding & Scoped Vector Storage, DeepSeek Generation & Citations, và Benchmark Evaluation.

**Architecture:** Kiến trúc Module hóa theo chuẩn Black-box và Code-agnostic. Luồng dữ liệu tuần tự: PDF Ingestion kiểm tra text layer (chặn scan) -> Flat-window chunking có gán siêu dữ liệu trang/workspace -> Vector hóa qua Jina API và lưu trữ vector có bộ lọc workspace -> Truy xuất tương đồng Cosine kết hợp DeepSeek v4 Flash sinh câu trả lời có trích dẫn -> Đánh giá hiệu năng tự động.

**Tech Stack:** Python 3.10+, `pydantic`, `pypdf`, `httpx`, `numpy`, `pytest`, `pytest-asyncio`.

**Spec:** `docs/superpowers/specs/2026-08-31-ebook-rag-baseline-hld.md`

## Global Constraints

- **Document Format**: Digital PDF tiếng Anh có text layer. Từ chối PDF scan với mã lỗi `ERR_UNSUPPORTED_SCANNED_PDF`.
- **Chunking**: Cửa sổ trượt cố định (Flat-Window) kích thước 512 tokens (hoặc ~400 từ), gối đầu 50 tokens (~10%).
- **Embedding Model**: `jina-embeddings-v5-omni-small` qua Jina AI API.
- **LLM Model**: `deepseek/deepseek-v4-flash-0731` qua OpenRouter API (`temperature=0.1`, `max_tokens=1024`).
- **Isolation**: Mọi truy vấn và lưu trữ phải được lọc cô lập theo `workspace_id`.
- **Citation**: Bắt buộc kèm Tên sách (`book_title`) và Số trang (`page_number`).

---

### Task 1: Core Domain Models & Configuration

**Files:**
- Create: `src/ebook_rag/core/models.py`
- Create: `src/ebook_rag/core/config.py`
- Test: `tests/core/test_models.py`

**Interfaces:**
- Produces: `Workspace`, `BookMetadata`, `Chunk`, `QueryResult`, `Citation`, `IngestionResult`, `RAGConfig`

- [ ] **Step 1: Write the failing test**

```python
# tests/core/test_models.py
import pytest
from src/ebook_rag/core/models import (
    Workspace,
    BookMetadata,
    Chunk,
    Citation,
    QueryResult,
    IngestionResult,
    IngestionStatus,
)
from src/ebook_rag/core/config import RAGConfig

def test_workspace_creation():
    ws = Workspace(workspace_id="ws_01", name="Software Engineering", description="SE Books")
    assert ws.workspace_id == "ws_01"
    assert ws.name == "Software Engineering"

def test_chunk_metadata_binding():
    chunk = Chunk(
        chunk_id="chk_01",
        book_id="bk_01",
        book_title="Clean Code",
        workspace_id="ws_01",
        page_number=15,
        text_content="Functions should do one thing.",
        chunk_index=0
    )
    assert chunk.page_number == 15
    assert chunk.workspace_id == "ws_01"

def test_ingestion_result_rejected_scan():
    res = IngestionResult(
        status=IngestionStatus.REJECTED,
        error_code="ERR_UNSUPPORTED_SCANNED_PDF",
        message="Scanned PDF format without valid text layer is not supported in MVP baseline."
    )
    assert res.status == IngestionStatus.REJECTED
    assert res.error_code == "ERR_UNSUPPORTED_SCANNED_PDF"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/core/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/ebook_rag/core/config.py
from pydantic import BaseModel

class RAGConfig(BaseModel):
    chunk_size: int = 512
    chunk_overlap: int = 50
    min_text_density_per_page: int = 50
    embedding_model: str = "jina-embeddings-v5-omni-small"
    embedding_batch_size: int = 32
    llm_model: str = "deepseek/deepseek-v4-flash-0731"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024
    top_k: int = 5
    jina_api_key: str = ""
    openrouter_api_key: str = ""
```

```python
# src/ebook_rag/core/models.py
from datetime import datetime
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field

class IngestionStatus(str, Enum):
    COMPLETED = "COMPLETED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"

class Workspace(BaseModel):
    workspace_id: str
    name: str
    description: Optional[str] = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)

class BookMetadata(BaseModel):
    book_id: str
    workspace_id: str
    title: str
    total_pages: int
    file_format: str = "PDF"

class Chunk(BaseModel):
    chunk_id: str
    book_id: str
    book_title: str
    workspace_id: str
    page_number: int
    text_content: str
    token_count: int = 0
    chunk_index: int

class Citation(BaseModel):
    book_title: str
    page_number: int
    relevance_score: float

class QueryResult(BaseModel):
    query: str
    workspace_id: str
    answer: str
    citations: List[Citation]
    retrieval_time_ms: float
    generation_time_ms: float
    total_latency_ms: float

class IngestionResult(BaseModel):
    status: IngestionStatus
    book_id: Optional[str] = None
    book_title: Optional[str] = None
    total_pages: Optional[int] = None
    total_chunks_indexed: Optional[int] = None
    processing_time_sec: Optional[float] = None
    error_code: Optional[str] = None
    message: Optional[str] = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/core/test_models.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ebook_rag/core/ tests/core/
git commit -m "feat: add core domain models and configuration"
```

---

### Task 2: Workspace Management Module

**Files:**
- Create: `src/ebook_rag/workspace/manager.py`
- Test: `tests/workspace/test_manager.py`

**Interfaces:**
- Consumes: `Workspace`
- Produces: `WorkspaceManager.create_workspace()`, `WorkspaceManager.list_workspaces()`, `WorkspaceManager.get_workspace()`, `WorkspaceManager.assign_book()`

- [ ] **Step 1: Write the failing test**

```python
# tests/workspace/test_manager.py
import pytest
from src/ebook_rag/workspace/manager import WorkspaceManager

def test_workspace_manager_crud():
    mgr = WorkspaceManager()
    ws = mgr.create_workspace(name="AI Engineering", description="LLM & RAG Books")
    assert ws.workspace_id is not None
    assert ws.name == "AI Engineering"

    all_ws = mgr.list_workspaces()
    assert len(all_ws) == 1
    assert all_ws[0].workspace_id == ws.workspace_id

    fetched = mgr.get_workspace(ws.workspace_id)
    assert fetched is not None
    assert fetched.name == "AI Engineering"

def test_workspace_book_assignment():
    mgr = WorkspaceManager()
    ws = mgr.create_workspace(name="Systems")
    success = mgr.assign_book(workspace_id=ws.workspace_id, book_id="bk_001")
    assert success is True

    books = mgr.list_workspace_books(ws.workspace_id)
    assert "bk_001" in books
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/workspace/test_manager.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/ebook_rag/workspace/manager.py
import uuid
from typing import Dict, List, Optional, Set
from src.ebook_rag.core.models import Workspace

class WorkspaceManager:
    def __init__(self):
        self._workspaces: Dict[str, Workspace] = {}
        self._workspace_books: Dict[str, Set[str]] = {}

    def create_workspace(self, name: str, description: str = "") -> Workspace:
        ws_id = f"ws_{uuid.uuid4().hex[:8]}"
        ws = Workspace(workspace_id=ws_id, name=name, description=description)
        self._workspaces[ws_id] = ws
        self._workspace_books[ws_id] = set()
        return ws

    def get_workspace(self, workspace_id: str) -> Optional[Workspace]:
        return self._workspaces.get(workspace_id)

    def list_workspaces(self) -> List[Workspace]:
        return list(self._workspaces.values())

    def assign_book(self, workspace_id: str, book_id: str) -> bool:
        if workspace_id not in self._workspaces:
            return False
        self._workspace_books[workspace_id].add(book_id)
        return True

    def list_workspace_books(self, workspace_id: str) -> List[str]:
        return list(self._workspace_books.get(workspace_id, set()))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/workspace/test_manager.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ebook_rag/workspace/ tests/workspace/
git commit -m "feat: add workspace manager module"
```

---

### Task 3: Phase 1 - PDF Ingestion, Text Density Validation & Flat-Window Chunking

**Files:**
- Create: `src/ebook_rag/ingestion/validator.py`
- Create: `src/ebook_rag/ingestion/chunker.py`
- Create: `src/ebook_rag/ingestion/extractor.py`
- Test: `tests/ingestion/test_ingestion_pipeline.py`

**Interfaces:**
- Consumes: `RAGConfig`, `Chunk`, `IngestionResult`
- Produces: `PDFValidator.validate_pdf()`, `FlatWindowChunker.chunk_pages()`, `PDFExtractor.extract_pages()`

- [ ] **Step 1: Write the failing test**

```python
# tests/ingestion/test_ingestion_pipeline.py
import pytest
from src/ebook_rag/ingestion/validator import PDFValidator
from src/ebook_rag/ingestion/chunker import FlatWindowChunker
from src/ebook_rag/core/config import RAGConfig

def test_validator_rejects_empty_or_scanned_pdf():
    config = RAGConfig(min_text_density_per_page=50)
    validator = PDFValidator(config)

    # Empty text page simulation (Scanned PDF)
    pages_text = ["", "   ", "\n\n"]
    is_valid, error = validator.validate_text_layer(pages_text)
    assert is_valid is False
    assert error == "ERR_UNSUPPORTED_SCANNED_PDF"

def test_validator_accepts_valid_digital_pdf():
    config = RAGConfig(min_text_density_per_page=50)
    validator = PDFValidator(config)

    pages_text = [
        "This is a digital textbook chapter on distributed systems with plenty of characters.",
        "Second page of the book discussing consensus algorithms such as Raft and Paxos."
    ]
    is_valid, error = validator.validate_text_layer(pages_text)
    assert is_valid is True
    assert error is None

def test_flat_window_chunker():
    config = RAGConfig(chunk_size=10, chunk_overlap=2)
    chunker = FlatWindowChunker(config)

    pages = [
        (1, "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10 word11 word12"),
        (2, "word13 word14 word15 word16 word17 word18 word19 word20")
    ]
    chunks = chunker.chunk_pages(
        pages=pages,
        workspace_id="ws_01",
        book_id="bk_01",
        book_title="Test Book"
    )

    assert len(chunks) > 1
    assert chunks[0].book_id == "bk_01"
    assert chunks[0].workspace_id == "ws_01"
    assert chunks[0].page_number in [1, 2]
    assert len(chunks[0].text_content) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/ingestion/test_ingestion_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/ebook_rag/ingestion/validator.py
from typing import List, Tuple, Optional
from src.ebook_rag.core.config import RAGConfig

class PDFValidator:
    def __init__(self, config: RAGConfig):
        self.min_density = config.min_text_density_per_page

    def validate_text_layer(self, pages_text: List[str]) -> Tuple[bool, Optional[str]]:
        if not pages_text:
            return False, "ERR_UNSUPPORTED_SCANNED_PDF"
        
        valid_pages_count = 0
        for text in pages_text:
            clean_text = "".join(text.split())
            if len(clean_text) >= self.min_density:
                valid_pages_count += 1

        # If more than 80% of pages have less than minimum density, consider it scanned
        if valid_pages_count / len(pages_text) < 0.2:
            return False, "ERR_UNSUPPORTED_SCANNED_PDF"
        
        return True, None
```

```python
# src/ebook_rag/ingestion/chunker.py
import uuid
from typing import List, Tuple
from src.ebook_rag.core.models import Chunk
from src.ebook_rag.core.config import RAGConfig

class FlatWindowChunker:
    def __init__(self, config: RAGConfig):
        self.chunk_size = config.chunk_size
        self.chunk_overlap = config.chunk_overlap

    def chunk_pages(
        self,
        pages: List[Tuple[int, str]],
        workspace_id: str,
        book_id: str,
        book_title: str
    ) -> List[Chunk]:
        chunks: List[Chunk] = []
        chunk_idx = 0

        for page_num, page_text in pages:
            words = page_text.strip().split()
            if not words:
                continue

            start = 0
            while start < len(words):
                end = min(start + self.chunk_size, len(words))
                chunk_words = words[start:end]
                text_content = " ".join(chunk_words)

                chunk = Chunk(
                    chunk_id=f"chk_{uuid.uuid4().hex[:10]}",
                    book_id=book_id,
                    book_title=book_title,
                    workspace_id=workspace_id,
                    page_number=page_num,
                    text_content=text_content,
                    token_count=len(chunk_words),
                    chunk_index=chunk_idx
                )
                chunks.append(chunk)
                chunk_idx += 1

                if end == len(words):
                    break
                start += max(1, self.chunk_size - self.chunk_overlap)

        return chunks
```

```python
# src/ebook_rag/ingestion/extractor.py
import io
from typing import List, Tuple
import pypdf

class PDFExtractor:
    @staticmethod
    def extract_pages(pdf_bytes: bytes) -> List[Tuple[int, str]]:
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        pages: List[Tuple[int, str]] = []
        for idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            pages.append((idx + 1, text))
        return pages
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/ingestion/test_ingestion_pipeline.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ebook_rag/ingestion/ tests/ingestion/
git commit -m "feat: add Phase 1 PDF validation and flat-window chunking"
```

---

### Task 4: Phase 2 - Jina Embedding Client & Scoped Vector Storage

**Files:**
- Create: `src/ebook_rag/embedding/jina_client.py`
- Create: `src/ebook_rag/storage/vector_store.py`
- Test: `tests/embedding_storage/test_embedding_storage.py`

**Interfaces:**
- Consumes: `RAGConfig`, `Chunk`
- Produces: `JinaEmbeddingClient.embed_texts()`, `ScopedVectorStore.add_chunks()`, `ScopedVectorStore.search_scoped()`

- [ ] **Step 1: Write the failing test**

```python
# tests/embedding_storage/test_embedding_storage.py
import pytest
import numpy as np
from src/ebook_rag/storage/vector_store import ScopedVectorStore
from src/ebook_rag/core/models import Chunk

def test_scoped_vector_store_filtering_and_similarity():
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

    # Search in ws_1: should NEVER return chunk from ws_2
    query_vec = [0.9, 0.1, 0.0]
    results = store.search_scoped(workspace_id="ws_1", query_vector=query_vec, top_k=5)

    assert len(results) == 1
    assert results[0][0].chunk_id == "c1"
    assert results[0][0].workspace_id == "ws_1"
    assert results[0][1] > 0.8  # Cosine similarity score
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/embedding_storage/test_embedding_storage.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/ebook_rag/storage/vector_store.py
from typing import List, Tuple, Dict
import numpy as np
from src.ebook_rag.core.models import Chunk

class ScopedVectorStore:
    def __init__(self):
        # workspace_id -> list of (Chunk, np.ndarray)
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
```

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

Run: `pytest tests/embedding_storage/test_embedding_storage.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ebook_rag/embedding/ src/ebook_rag/storage/ tests/embedding_storage/
git commit -m "feat: add Phase 2 Jina embedding client and scoped vector store"
```

---

### Task 5: Phase 3 - DeepSeek Generation & Citation Formatter

**Files:**
- Create: `src/ebook_rag/generation/deepseek_client.py`
- Test: `tests/generation/test_deepseek_generation.py`

**Interfaces:**
- Consumes: `RAGConfig`, `Chunk`, `Citation`
- Produces: `DeepSeekGenerator.generate_response()`

- [ ] **Step 1: Write the failing test**

```python
# tests/generation/test_deepseek_generation.py
import pytest
from src/ebook_rag/generation/deepseek_client import DeepSeekGenerator
from src/ebook_rag/core/models import Chunk
from src/ebook_rag/core/config import RAGConfig

def test_prompt_context_assembly():
    config = RAGConfig()
    generator = DeepSeekGenerator(config)

    chunks_with_scores = [
        (
            Chunk(
                chunk_id="c1", book_id="b1", book_title="Clean Code",
                workspace_id="ws_1", page_number=45, text_content="Names should reveal intent.",
                chunk_index=0
            ),
            0.92
        )
    ]

    prompt = generator.build_augmented_prompt("How should we name variables?", chunks_with_scores)
    assert "Clean Code" in prompt
    assert "Page: 45" in prompt
    assert "Names should reveal intent." in prompt
    assert "How should we name variables?" in prompt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/generation/test_deepseek_generation.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/ebook_rag/generation/deepseek_client.py
from typing import List, Tuple
import httpx
from src.ebook_rag.core.models import Chunk, Citation
from src.ebook_rag.core.config import RAGConfig

class DeepSeekGenerator:
    def __init__(self, config: RAGConfig):
        self.model = config.llm_model
        self.temperature = config.llm_temperature
        self.max_tokens = config.llm_max_tokens
        self.api_key = config.openrouter_api_key
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"

    def build_augmented_prompt(
        self,
        query: str,
        retrieved_chunks: List[Tuple[Chunk, float]]
    ) -> str:
        context_blocks = []
        for chunk, score in retrieved_chunks:
            context_blocks.append(
                f"[Document: {chunk.book_title} | Page: {chunk.page_number}]\n{chunk.text_content}"
            )
        
        context_str = "\n\n---\n\n".join(context_blocks)
        prompt = (
            "You are an expert E-Book assistant. Answer the user's question strictly using the provided context below.\n"
            "If the answer cannot be deduced from the context, state clearly that the information is not present.\n\n"
            f"Context:\n{context_str}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )
        return prompt

    async def generate_response(
        self,
        query: str,
        retrieved_chunks: List[Tuple[Chunk, float]]
    ) -> Tuple[str, List[Citation]]:
        if not retrieved_chunks:
            return "No relevant context found in this workspace.", []

        prompt = self.build_augmented_prompt(query, retrieved_chunks)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

        async with httpx.AsyncClient(timeout=45.0) as client:
            resp = await client.post(self.api_url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            answer = data["choices"][0]["message"]["content"].strip()

        citations: List[Citation] = [
            Citation(
                book_title=chunk.book_title,
                page_number=chunk.page_number,
                relevance_score=round(score, 4)
            )
            for chunk, score in retrieved_chunks
        ]

        return answer, citations
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/generation/test_deepseek_generation.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ebook_rag/generation/ tests/generation/
git commit -m "feat: add Phase 3 DeepSeek generator and citation formatter"
```

---

### Task 6: Phase 3 - End-to-End EBookRAGPipeline Orchestration

**Files:**
- Create: `src/ebook_rag/pipeline.py`
- Test: `tests/test_e2e_pipeline.py`

**Interfaces:**
- Consumes: `WorkspaceManager`, `PDFValidator`, `PDFExtractor`, `FlatWindowChunker`, `JinaEmbeddingClient`, `ScopedVectorStore`, `DeepSeekGenerator`
- Produces: `EBookRAGPipeline.ingest_book()`, `EBookRAGPipeline.query_workspace()`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_e2e_pipeline.py
import pytest
from unittest.mock import AsyncMock, patch
from src/ebook_rag/pipeline import EBookRAGPipeline
from src/ebook_rag/core.config import RAGConfig
from src/ebook_rag/core.models import IngestionStatus

@pytest.mark.asyncio
async def test_pipeline_ingest_and_query_flow():
    config = RAGConfig(min_text_density_per_page=10)
    pipeline = EBookRAGPipeline(config)

    # 1. Create Workspace
    ws = pipeline.workspace_mgr.create_workspace(name="DevOps")

    # 2. Mock Embedding & Generation
    mock_embeddings = [[0.1, 0.2, 0.3]]
    pipeline.embedding_client.embed_texts = AsyncMock(return_value=mock_embeddings)
    pipeline.generator.generate_response = AsyncMock(
        return_value=("Continuous Integration is the practice of...", [])
    )

    # 3. Simulate Ingestion with mock PDF pages
    with patch("src.ebook_rag.ingestion.extractor.PDFExtractor.extract_pages", return_value=[
        (1, "Continuous Integration is the practice of automating the integration of code changes.")
    ]):
        ingest_res = await pipeline.ingest_book(
            pdf_bytes=b"%PDF-1.4 mock",
            workspace_id=ws.workspace_id,
            book_title="CI/CD Handbook"
        )
        assert ingest_res.status == IngestionStatus.COMPLETED
        assert ingest_res.total_pages == 1
        assert ingest_res.total_chunks_indexed == 1

    # 4. Query
    query_res = await pipeline.query_workspace(
        workspace_id=ws.workspace_id,
        query_text="What is Continuous Integration?"
    )
    assert "Continuous Integration" in query_res.answer
    assert query_res.workspace_id == ws.workspace_id
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_e2e_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/ebook_rag/pipeline.py
import time
import uuid
from typing import List, Optional
from src.ebook_rag.core.config import RAGConfig
from src.ebook_rag.core.models import (
    IngestionResult,
    IngestionStatus,
    QueryResult,
    BookMetadata
)
from src.ebook_rag.workspace.manager import WorkspaceManager
from src.ebook_rag.ingestion.validator import PDFValidator
from src.ebook_rag.ingestion.extractor import PDFExtractor
from src.ebook_rag.ingestion.chunker import FlatWindowChunker
from src.ebook_rag.embedding.jina_client import JinaEmbeddingClient
from src.ebook_rag.storage.vector_store import ScopedVectorStore
from src.ebook_rag.generation.deepseek_client import DeepSeekGenerator

class EBookRAGPipeline:
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        self.workspace_mgr = WorkspaceManager()
        self.validator = PDFValidator(self.config)
        self.chunker = FlatWindowChunker(self.config)
        self.embedding_client = JinaEmbeddingClient(self.config)
        self.vector_store = ScopedVectorStore()
        self.generator = DeepSeekGenerator(self.config)

    async def ingest_book(
        self,
        pdf_bytes: bytes,
        workspace_id: str,
        book_title: str
    ) -> IngestionResult:
        start_time = time.time()
        
        # Check workspace existence
        if not self.workspace_mgr.get_workspace(workspace_id):
            return IngestionResult(
                status=IngestionStatus.FAILED,
                error_code="ERR_WORKSPACE_NOT_FOUND",
                message=f"Workspace {workspace_id} does not exist."
            )

        # 1. Extract pages
        try:
            pages = PDFExtractor.extract_pages(pdf_bytes)
        except Exception as e:
            return IngestionResult(
                status=IngestionStatus.FAILED,
                error_code="ERR_PDF_EXTRACTION_FAILED",
                message=str(e)
            )

        # 2. Validate text layer (Check for Scanned PDF)
        pages_text = [p[1] for p in pages]
        is_valid, error_code = self.validator.validate_text_layer(pages_text)
        if not is_valid:
            return IngestionResult(
                status=IngestionStatus.REJECTED,
                error_code=error_code,
                message="Scanned PDF format without valid text layer is not supported in MVP baseline."
            )

        # 3. Flat-Window Chunking
        book_id = f"bk_{uuid.uuid4().hex[:8]}"
        chunks = self.chunker.chunk_pages(
            pages=pages,
            workspace_id=workspace_id,
            book_id=book_id,
            book_title=book_title
        )

        if not chunks:
            return IngestionResult(
                status=IngestionStatus.FAILED,
                error_code="ERR_NO_CHUNKS_GENERATED",
                message="No chunks could be extracted."
            )

        # 4. Embed Chunks
        chunk_texts = [c.text_content for c in chunks]
        vectors = await self.embedding_client.embed_texts(chunk_texts)

        # 5. Index into Scoped Vector Store
        self.vector_store.add_chunks(chunks, vectors)
        self.workspace_mgr.assign_book(workspace_id, book_id)

        duration = time.time() - start_time
        return IngestionResult(
            status=IngestionStatus.COMPLETED,
            book_id=book_id,
            book_title=book_title,
            total_pages=len(pages),
            total_chunks_indexed=len(chunks),
            processing_time_sec=round(duration, 2)
        )

    async def query_workspace(
        self,
        workspace_id: str,
        query_text: str,
        top_k: Optional[int] = None
    ) -> QueryResult:
        start_time = time.time()
        k = top_k or self.config.top_k

        # 1. Embed query
        retrieval_start = time.time()
        query_embeddings = await self.embedding_client.embed_texts([query_text])
        query_vec = query_embeddings[0] if query_embeddings else [0.0] * 1024

        # 2. Scoped Retrieval
        retrieved_chunks = self.vector_store.search_scoped(
            workspace_id=workspace_id,
            query_vector=query_vec,
            top_k=k
        )
        retrieval_time_ms = (time.time() - retrieval_start) * 1000

        # 3. Generation
        gen_start = time.time()
        answer, citations = await self.generator.generate_response(
            query=query_text,
            retrieved_chunks=retrieved_chunks
        )
        gen_time_ms = (time.time() - gen_start) * 1000

        total_latency_ms = (time.time() - start_time) * 1000
        return QueryResult(
            query=query_text,
            workspace_id=workspace_id,
            answer=answer,
            citations=citations,
            retrieval_time_ms=round(retrieval_time_ms, 2),
            generation_time_ms=round(gen_time_ms, 2),
            total_latency_ms=round(total_latency_ms, 2)
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_e2e_pipeline.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ebook_rag/pipeline.py tests/test_e2e_pipeline.py
git commit -m "feat: add end-to-end EBookRAGPipeline orchestrator"
```

---

### Task 7: Phase 4 - Evaluation Engine & Baseline Benchmark Suite

**Files:**
- Create: `src/ebook_rag/evaluation/benchmark.py`
- Test: `tests/evaluation/test_benchmark.py`

**Interfaces:**
- Consumes: `EBookRAGPipeline`, `QueryResult`
- Produces: `BenchmarkRunner.run_evaluation()`, `BenchmarkReport`

- [ ] **Step 1: Write the failing test**

```python
# tests/evaluation/test_benchmark.py
import pytest
from unittest.mock import AsyncMock
from src/ebook_rag/evaluation/benchmark import BenchmarkRunner, TestCase
from src/ebook_rag/pipeline import EBookRAGPipeline
from src/ebook_rag/core.models import QueryResult, Citation

@pytest.mark.asyncio
async def test_benchmark_runner_metrics():
    pipeline = EBookRAGPipeline()
    pipeline.query_workspace = AsyncMock(return_value=QueryResult(
        query="What is SRP?",
        workspace_id="ws_1",
        answer="SRP states a class should have one reason to change.",
        citations=[Citation(book_title="Clean Code", page_number=138, relevance_score=0.9)],
        retrieval_time_ms=100.0,
        generation_time_ms=500.0,
        total_latency_ms=600.0
    ))

    test_cases = [
        TestCase(
            query="What is SRP?",
            expected_page=138,
            expected_keywords=["reason to change"]
        )
    ]

    runner = BenchmarkRunner(pipeline)
    report = await runner.run_evaluation("ws_1", test_cases)

    assert report.total_test_cases == 1
    assert report.recall_at_k == 1.0
    assert report.faithfulness_rate >= 0.8
    assert report.avg_latency_ms == 600.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/evaluation/test_benchmark.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/ebook_rag/evaluation/benchmark.py
from typing import List, Optional
from pydantic import BaseModel
from src.ebook_rag.pipeline import EBookRAGPipeline

class TestCase(BaseModel):
    query: str
    expected_page: Optional[int] = None
    expected_keywords: List[str] = []

class BenchmarkReport(BaseModel):
    total_test_cases: int
    recall_at_k: float
    faithfulness_rate: float
    avg_latency_ms: float
    passed_threshold: bool

class BenchmarkRunner:
    def __init__(self, pipeline: EBookRAGPipeline):
        self.pipeline = pipeline

    async def run_evaluation(
        self,
        workspace_id: str,
        test_cases: List[TestCase]
    ) -> BenchmarkReport:
        if not test_cases:
            return BenchmarkReport(
                total_test_cases=0,
                recall_at_k=0.0,
                faithfulness_rate=0.0,
                avg_latency_ms=0.0,
                passed_threshold=False
            )

        correct_retrievals = 0
        faithful_answers = 0
        total_latency = 0.0

        for tc in test_cases:
            res = await self.pipeline.query_workspace(workspace_id, tc.query)
            total_latency += res.total_latency_ms

            # Recall@K check
            if tc.expected_page is not None:
                retrieved_pages = [c.page_number for c in res.citations]
                if tc.expected_page in retrieved_pages:
                    correct_retrievals += 1
            else:
                correct_retrievals += 1

            # Faithfulness check based on keyword evidence
            if tc.expected_keywords:
                matched = any(kw.lower() in res.answer.lower() for kw in tc.expected_keywords)
                if matched:
                    faithful_answers += 1
            else:
                faithful_answers += 1

        n = len(test_cases)
        recall = correct_retrievals / n
        faithfulness = faithful_answers / n
        avg_lat = total_latency / n

        # Target criteria: Recall >= 0.70, Faithfulness >= 0.85
        passed = (recall >= 0.70) and (faithfulness >= 0.85)

        return BenchmarkReport(
            total_test_cases=n,
            recall_at_k=round(recall, 4),
            faithfulness_rate=round(faithfulness, 4),
            avg_latency_ms=round(avg_lat, 2),
            passed_threshold=passed
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/evaluation/test_benchmark.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ebook_rag/evaluation/ tests/evaluation/
git commit -m "feat: add Phase 4 benchmark runner and evaluation metrics"
```
