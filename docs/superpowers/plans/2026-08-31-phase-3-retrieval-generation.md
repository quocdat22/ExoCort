# Phase 3: Scoped Retrieval & Generation Implementation Plan (Managed with uv)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Xây dựng Phase 3 bao gồm bộ máy tạo Prompt ngữ cảnh, tích hợp mô hình `deepseek/deepseek-v4-flash-0731` qua OpenRouter để sinh câu trả lời kèm trích dẫn chuẩn, và hợp nhất toàn bộ luồng xử lý thành `EBookRAGPipeline`. Môi trường thực thi quản lý bởi `uv`.

**Architecture:** Nhận truy vấn và `workspace_id` -> Vector hóa câu hỏi qua Jina -> Truy xuất Top-K từ `ScopedVectorStore` -> Đóng gói khối ngữ cảnh -> Gửi tới DeepSeek v4 Flash với quy tắc trích dẫn nghiêm ngặt -> Trả về `QueryResult` hoàn chỉnh.

**Tech Stack:** `uv`, Python 3.10+, `httpx`, `pytest`, `pytest-asyncio`, `tenacity`.

**Spec:** `docs/superpowers/specs/2026-08-31-ebook-rag-baseline-hld.md`

## Global Constraints

- **Environment & Execution**: Toàn bộ các lệnh chạy test, cài đặt dependencies và thực thi phải thông qua **`uv`** (ví dụ: `uv run pytest`).
- **Input Hand-off**: Nhận `ScopedVectorStore` và `JinaEmbeddingClient` từ Phase 2, `PDFValidator` và `FlatWindowChunker` từ Phase 1. Tất cả được import từ `exocort`.
- **LLM Model**: `deepseek/deepseek-v4-flash-0731` qua OpenRouter API.
- **Citation Format**: Bắt buộc gắn kèm `book_title` cùng với `page_start` và `page_end`.
- **Latency**: Tối ưu độ trễ tổng thể $\le 2.5$ giây.
- **Output Hand-off**: Cung cấp `EBookRAGPipeline` hoàn chỉnh sẵn sàng cho việc đánh giá ở Phase 4.

---

### Task 1: DeepSeek Generation Client & Citation Formatter

**Files:**
- Create: `src/exocort/generation/deepseek_client.py`
- Test: `tests/generation/test_deepseek_generation.py`

**Interfaces:**
- Consumes: `RAGConfig`, `Chunk`, `Citation`
- Produces: `DeepSeekGenerator.build_augmented_prompt()`, `DeepSeekGenerator.generate_response()`

- [ ] **Step 1: Write failing unit test**

```python
# tests/generation/test_deepseek_generation.py
import pytest
from unittest.mock import AsyncMock, patch
from exocort.generation.deepseek_client import DeepSeekGenerator
from exocort.core.models import Chunk
from exocort.core.config import RAGConfig

def test_prompt_context_assembly():
    config = RAGConfig()
    generator = DeepSeekGenerator(config)

    chunks_with_scores = [
        (
            Chunk(
                chunk_id="c1", book_id="b1", book_title="Clean Code",
                workspace_id="ws_1", page_start=45, page_end=45, text_content="Names should reveal intent.",
                chunk_index=0
            ),
            0.92
        ),
        (
            Chunk(
                chunk_id="c2", book_id="b2", book_title="Pragmatic Programmer",
                workspace_id="ws_1", page_start=10, page_end=11, text_content="Don't repeat yourself.",
                chunk_index=1
            ),
            0.85
        )
    ]

    prompt = generator.build_augmented_prompt("How should we name variables?", chunks_with_scores)
    assert "Clean Code" in prompt
    assert "Page: 45" in prompt
    assert "Names should reveal intent." in prompt
    assert "Pragmatic Programmer" in prompt
    assert "Pages: 10-11" in prompt
    assert "How should we name variables?" in prompt

@pytest.mark.asyncio
async def test_deepseek_generate_response_mock():
    config = RAGConfig(openrouter_api_key="mock_key")
    generator = DeepSeekGenerator(config)

    mock_resp_payload = {
        "choices": [{
            "message": {
                "content": "Variable names should be descriptive and reveal intent."
            }
        }]
    }

    chunks_with_scores = [
        (
            Chunk(
                chunk_id="c1", book_id="b1", book_title="Clean Code",
                workspace_id="ws_1", page_start=45, page_end=45, text_content="Names should reveal intent.",
                chunk_index=0
            ),
            0.92
        )
    ]

    with patch("httpx.AsyncClient.post") as mock_post:
        mock_post.return_value = AsyncMock(
            json=lambda: mock_resp_payload,
            raise_for_status=lambda: None
        )

        answer, citations = await generator.generate_response("How to name?", chunks_with_scores)
        assert "Variable names should be descriptive" in answer
        assert len(citations) == 1
        assert citations[0].book_title == "Clean Code"
        assert citations[0].page_start == 45
        assert citations[0].page_end == 45
        assert citations[0].text_content == "Names should reveal intent."
```

- [ ] **Step 2: Run test using uv to verify it fails**

Run: `uv run pytest tests/generation/test_deepseek_generation.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/exocort/generation/deepseek_client.py
from typing import List, Tuple
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
from exocort.core.models import Chunk, Citation
from exocort.core.config import RAGConfig

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
            if chunk.page_start == chunk.page_end:
                context_blocks.append(
                    f"[Document: {chunk.book_title} | Page: {chunk.page_start}]\n{chunk.text_content}"
                )
            else:
                context_blocks.append(
                    f"[Document: {chunk.book_title} | Pages: {chunk.page_start}-{chunk.page_end}]\n{chunk.text_content}"
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

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=30))
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
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                relevance_score=round(score, 4),
                text_content=chunk.text_content,
            )
            for chunk, score in retrieved_chunks
        ]

        return answer, citations
```

- [ ] **Step 4: Run test using uv to verify it passes**

Run: `uv run pytest tests/generation/test_deepseek_generation.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/exocort/generation/ tests/generation/
git commit -m "feat(phase-3): add DeepSeek generator and citation formatter for exocort"
```

---

### Task 2: End-to-End EBookRAGPipeline Orchestrator

**Files:**
- Create: `src/exocort/pipeline.py`
- Test: `tests/test_e2e_pipeline.py`

**Interfaces:**
- Consumes: All Phase 1, 2, and 3 modules
- Produces: `EBookRAGPipeline.ingest_book()`, `EBookRAGPipeline.query_workspace()`

- [ ] **Step 1: Write failing integration test**

```python
# tests/test_e2e_pipeline.py
import pytest
import tempfile
from unittest.mock import AsyncMock, patch
from exocort.pipeline import EBookRAGPipeline
from exocort.core.config import RAGConfig
from exocort.core.models import IngestionStatus

@pytest.mark.asyncio
async def test_pipeline_ingest_and_query_flow():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = RAGConfig(min_text_density_per_page=10, chroma_dir=tmpdir)
        pipeline = EBookRAGPipeline(config)

        # 1. Create Workspace
        ws = pipeline.workspace_mgr.create_workspace(name="DevOps")

        # 2. Mock Embedding & Generation
        mock_embeddings = [[0.1, 0.2, 0.3]]
        pipeline.embedding_client.embed_texts = AsyncMock(return_value=mock_embeddings)
        pipeline.generator.generate_response = AsyncMock(
            return_value=("Continuous Integration is the practice of...", [])
        )

        # 3. Ingestion
        with patch("exocort.ingestion.extractor.PDFExtractor.extract_pages", return_value=[
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

        # 4. Scoped Query
        query_res = await pipeline.query_workspace(
            workspace_id=ws.workspace_id,
            query_text="What is Continuous Integration?"
        )
        assert "Continuous Integration" in query_res.answer
        assert query_res.workspace_id == ws.workspace_id
        assert query_res.total_latency_ms > 0
```

- [ ] **Step 2: Run test using uv to verify it fails**

Run: `uv run pytest tests/test_e2e_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/exocort/pipeline.py
import time
import uuid
from typing import List, Optional
from exocort.core.config import RAGConfig
from exocort.core.models import (
    IngestionResult,
    IngestionStatus,
    QueryResult,
)
from exocort.workspace.manager import WorkspaceManager
from exocort.ingestion.validator import PDFValidator
from exocort.ingestion.extractor import PDFExtractor
from exocort.ingestion.chunker import FlatWindowChunker
from exocort.embedding.jina_client import JinaEmbeddingClient
from exocort.storage.vector_store import ScopedVectorStore
from exocort.generation.deepseek_client import DeepSeekGenerator

class EBookRAGPipeline:
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        self.workspace_mgr = WorkspaceManager()
        self.validator = PDFValidator(self.config)
        self.chunker = FlatWindowChunker(self.config)
        self.embedding_client = JinaEmbeddingClient(self.config)
        
        persist_dir = getattr(self.config, 'chroma_dir', './chroma_db')
        self.vector_store = ScopedVectorStore(persist_dir=persist_dir)
        
        self.generator = DeepSeekGenerator(self.config)

    async def ingest_book(
        self,
        pdf_bytes: bytes,
        workspace_id: str,
        book_title: str
    ) -> IngestionResult:
        start_time = time.time()
        
        # Verify workspace existence
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

        # 2. Validate text layer (Block Scanned PDF)
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

- [ ] **Step 4: Run test using uv to verify it passes**

Run: `uv run pytest tests/test_e2e_pipeline.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/exocort/pipeline.py tests/test_e2e_pipeline.py
git commit -m "feat(phase-3): add end-to-end EBookRAGPipeline orchestrator for exocort"
```
