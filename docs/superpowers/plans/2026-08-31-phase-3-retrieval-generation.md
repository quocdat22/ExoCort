# Phase 3: Scoped Retrieval & Generation Implementation Plan (Managed with uv)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Xây dựng Phase 3 bao gồm bộ máy tạo Prompt ngữ cảnh, tích hợp mô hình `deepseek/deepseek-v4-flash-0731` qua OpenRouter để sinh câu trả lời kèm trích dẫn chuẩn, và hợp nhất toàn bộ luồng xử lý thành `EBookRAGPipeline`. Môi trường thực thi quản lý bởi `uv`.

**Architecture:** Nhận truy vấn và `workspace_id` -> Vector hóa câu hỏi qua Jina -> Truy xuất Top-K từ `ScopedVectorStore` -> Đóng gói khối ngữ cảnh -> Gửi tới DeepSeek v4 Flash với quy tắc trích dẫn nghiêm ngặt -> Trả về `QueryResult` hoàn chỉnh.

**Tech Stack:** `uv`, Python 3.10+, `httpx`, `pytest`, `pytest-asyncio`, `tenacity`.

**Spec:** `docs/superpowers/specs/2026-08-31-exocort-baseline-hld.md`

## Global Constraints

- **Environment & Execution**: Toàn bộ các lệnh chạy test, cài đặt dependencies và thực thi phải thông qua **`uv`** (ví dụ: `uv run pytest`).
- **Input Hand-off**: Nhận `ScopedVectorStore` và `JinaEmbeddingClient` từ Phase 2, `PDFValidator` và `FlatWindowChunker` từ Phase 1. Tất cả được import từ `exocort`.
- **LLM Model**: `deepseek/deepseek-v4-flash-0731` qua OpenRouter API.
- **Citation Format**: Bắt buộc gắn kèm `book_title` cùng với `page_start` và `page_end`.
- **Latency**: Tối ưu độ trễ tổng thể $\le 2.5$ giây.
- **Output Hand-off**: Cung cấp `EBookRAGPipeline` hoàn chỉnh sẵn sàng cho việc đánh giá ở Phase 4.

---

### Task 1: DeepSeek Generation Client & Citation Formatter with Fast Agile Resilience

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
from pydantic import SecretStr
from exocort.generation.deepseek_client import DeepSeekGenerator
from exocort.core.models import Chunk
from exocort.core.config import RAGConfig

def test_prompt_context_assembly():
    config = RAGConfig(openrouter_api_key=SecretStr("mock_key"))
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

def test_deepseek_generator_missing_api_key():
    config = RAGConfig(openrouter_api_key=SecretStr(""))
    with pytest.raises(ValueError, match="ERR_MISSING_API_KEY"):
        DeepSeekGenerator(config)

@pytest.mark.asyncio
async def test_deepseek_generate_response_mock():
    config = RAGConfig(openrouter_api_key=SecretStr("mock_key"), query_generation_timeout=5.0)
    generator = DeepSeekGenerator(config)

    mock_resp_payload = {
        "choices": [{
            "message": {
                "content": "Variable names should be descriptive and reveal intent."
            }
        }],
        "usage": {
            "prompt_tokens": 150,
            "completion_tokens": 25,
            "total_tokens": 175
        }
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

        answer, citations, tokens = await generator.generate_response("How to name?", chunks_with_scores)
        assert "Variable names should be descriptive" in answer
        assert len(citations) == 1
        assert citations[0].book_title == "Clean Code"
        assert citations[0].page_start == 45
        assert citations[0].page_end == 45
        assert citations[0].text_content == "Names should reveal intent."
        assert tokens["total_tokens"] == 175
```

- [ ] **Step 2: Run test using uv to verify it fails**

Run: `uv run pytest tests/generation/test_deepseek_generation.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/exocort/generation/deepseek_client.py
from typing import Dict, List, Tuple
import httpx
from tenacity import retry, stop_after_attempt, wait_fixed
from exocort.core.models import Chunk, Citation
from exocort.core.config import RAGConfig

class DeepSeekGenerator:
    """Generation client using DeepSeek v4 Flash via OpenRouter with Fast Agile Resilience."""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.model = config.llm_model
        self.temperature = config.llm_temperature
        self.max_tokens = config.llm_max_tokens
        self.api_key = (
            config.openrouter_api_key.get_secret_value()
            if hasattr(config.openrouter_api_key, "get_secret_value")
            else str(config.openrouter_api_key)
        )
        if not self.api_key:
            raise ValueError("ERR_MISSING_API_KEY: OPENROUTER_API_KEY must be provided.")
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

    @retry(stop=stop_after_attempt(2), wait=wait_fixed(0.5))
    async def generate_response(
        self,
        query: str,
        retrieved_chunks: List[Tuple[Chunk, float]]
    ) -> Tuple[str, List[Citation], Dict[str, int]]:
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

        if not retrieved_chunks:
            return "No relevant context found in this workspace.", [], {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

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

        async with httpx.AsyncClient(timeout=self.config.query_generation_timeout) as client:
            resp = await client.post(self.api_url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            answer = data["choices"][0]["message"]["content"].strip()
            usage = data.get("usage", {})
            tokens = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }

        return answer, citations, tokens
```

- [ ] **Step 4: Run test using uv to verify it passes**

Run: `uv run pytest tests/generation/test_deepseek_generation.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/exocort/generation/ tests/generation/
git commit -m "feat(phase-3): add DeepSeek generator with fast agile resilience and token telemetry"
```

---

### Task 2: End-to-End EBookRAGPipeline Orchestrator with Resilience & Deduplication

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
from exocort.core.models import IngestionStatus, QueryStatus

@pytest.mark.asyncio
async def test_pipeline_ingest_and_query_flow():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = RAGConfig(
            min_text_density_per_page=10,
            chroma_persist_dir=tmpdir,
            jina_api_key=SecretStr("mock_jina_key"),
            openrouter_api_key=SecretStr("mock_openrouter_key"),
        )
        pipeline = EBookRAGPipeline(config)

        # 1. Create Workspace
        ws = pipeline.workspace_mgr.create_workspace(name="DevOps")

        # 2. Mock Embedding & Generation
        mock_embeddings = [[0.1] * 1024]
        pipeline.embedding_client.embed_texts = AsyncMock(return_value=mock_embeddings)
        pipeline.embedding_client.embed_query = AsyncMock(return_value=[0.1] * 1024)
        pipeline.generator.generate_response = AsyncMock(
            return_value=("Continuous Integration is automated.", [], {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
        )

        # 3. Ingestion
        with patch("exocort.ingestion.extractor.PDFExtractor.extract_pages", return_value=[
            (1, "Continuous Integration is automated.")
        ]):
            ingest_res = await pipeline.ingest_book(
                pdf_bytes=b"%PDF-1.4 mock",
                workspace_id=ws.workspace_id,
                book_title="CI/CD Handbook"
            )
            assert ingest_res.status == IngestionStatus.COMPLETED
            assert ingest_res.total_pages == 1
            assert ingest_res.total_chunks_indexed == 1

        # 4. Deduplication Check (overwrite=False)
        with patch("exocort.ingestion.extractor.PDFExtractor.extract_pages", return_value=[(1, "text")]):
            dup_res = await pipeline.ingest_book(
                pdf_bytes=b"%PDF-1.4 mock",
                workspace_id=ws.workspace_id,
                book_title="CI/CD Handbook",
                overwrite=False
            )
            assert dup_res.status == IngestionStatus.REJECTED
            assert dup_res.error_code == "ERR_DUPLICATE_BOOK_TITLE"

        # 5. Scoped Query
        query_res = await pipeline.query_workspace(
            workspace_id=ws.workspace_id,
            query_text="What is Continuous Integration?"
        )
        assert query_res.status == QueryStatus.SUCCESS
        assert "Continuous Integration" in query_res.answer
        assert query_res.workspace_id == ws.workspace_id
        assert query_res.total_latency_ms > 0

@pytest.mark.asyncio
async def test_pipeline_generation_failure_partial_fallback():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = RAGConfig(
            chroma_persist_dir=tmpdir,
            jina_api_key=SecretStr("mock_jina_key"),
            openrouter_api_key=SecretStr("mock_openrouter_key"),
        )
        pipeline = EBookRAGPipeline(config)
        ws = pipeline.workspace_mgr.create_workspace(name="AI")

        # Mock embedding succeed, generation fail
        pipeline.embedding_client.embed_query = AsyncMock(return_value=[0.1] * 1024)
        pipeline.generator.generate_response = AsyncMock(side_effect=RuntimeError("OpenRouter 503"))

        query_res = await pipeline.query_workspace(
            workspace_id=ws.workspace_id,
            query_text="Explain RAG"
        )
        assert query_res.status == QueryStatus.PARTIAL
        assert query_res.error_code == "ERR_UPSTREAM_GENERATION_FAILED"
        assert "tạm thời gián đoạn" in query_res.answer

def test_pipeline_init_missing_api_keys_fails_fast():
    config = RAGConfig(jina_api_key=SecretStr(""), openrouter_api_key=SecretStr(""))
    with pytest.raises(ValueError, match="ERR_MISSING_API_KEY"):
        EBookRAGPipeline(config)
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
    BookMetadata,
    Citation,
    IngestionResult,
    IngestionStatus,
    QueryResult,
    QueryStatus,
)
from exocort.workspace.manager import WorkspaceManager
from exocort.ingestion.validator import PDFValidator
from exocort.ingestion.extractor import PDFExtractor
from exocort.ingestion.chunker import FlatWindowChunker
from exocort.embedding.jina_client import JinaEmbeddingClient
from exocort.storage.vector_store import ScopedVectorStore
from exocort.generation.deepseek_client import DeepSeekGenerator

class EBookRAGPipeline:
    """Unified Orchestrator for E-Book RAG Engine Baseline MVP."""

    def __init__(self, config: Optional[RAGConfig] = None, validate_keys: bool = True):
        self.config = config or RAGConfig()
        if validate_keys:
            self.config.validate_api_keys()
        self.workspace_mgr = WorkspaceManager()
        self.validator = PDFValidator(self.config)
        self.chunker = FlatWindowChunker(self.config)
        self.embedding_client = JinaEmbeddingClient(self.config)
        self.vector_store = ScopedVectorStore(persist_dir=self.config.chroma_persist_dir)
        self.generator = DeepSeekGenerator(self.config)

    async def ingest_book(
        self,
        pdf_bytes: bytes,
        workspace_id: str,
        book_title: str,
        overwrite: bool = False
    ) -> IngestionResult:
        start_time = time.time()
        
        # 1. Verify workspace existence
        if not self.workspace_mgr.get_workspace(workspace_id):
            return IngestionResult(
                status=IngestionStatus.FAILED,
                error_code="ERR_WORKSPACE_NOT_FOUND",
                message=f"Workspace {workspace_id} does not exist."
            )

        # 2. Deduplication Check
        existing_book = self.workspace_mgr.get_book_by_title(workspace_id, book_title)
        if existing_book and not overwrite:
            return IngestionResult(
                status=IngestionStatus.REJECTED,
                error_code="ERR_DUPLICATE_BOOK_TITLE",
                message=f"Book with title '{book_title}' already exists in workspace '{workspace_id}'. Set overwrite=true to replace."
            )

        # 3. Extract pages
        try:
            pages = PDFExtractor.extract_pages(pdf_bytes)
        except Exception as e:
            return IngestionResult(
                status=IngestionStatus.FAILED,
                error_code="ERR_PDF_EXTRACTION_FAILED",
                message=f"Failed to extract PDF: {str(e)}"
            )

        # 4. Guardrails & Validation Gate
        is_valid_count, count_error = self.validator.validate_page_count(len(pages))
        if not is_valid_count:
            return IngestionResult(
                status=IngestionStatus.REJECTED,
                error_code=count_error,
                message=f"Document contains {len(pages)} pages, exceeding maximum limit of {self.config.max_pages_per_book}."
            )

        pages_text = [p[1] for p in pages]
        is_valid_text, text_error = self.validator.validate_text_layer(pages_text)
        if not is_valid_text:
            return IngestionResult(
                status=IngestionStatus.REJECTED,
                error_code=text_error,
                message="Scanned PDF format without valid text layer is not supported in MVP baseline."
            )

        # 5. Flat-Window Chunking
        book_id = existing_book.book_id if (existing_book and overwrite) else f"bk_{uuid.uuid4().hex[:8]}"
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
                message="No valid chunks could be extracted from the document."
            )

        if len(chunks) > self.config.max_chunks_per_book:
            return IngestionResult(
                status=IngestionStatus.REJECTED,
                error_code="ERR_DOCUMENT_TOO_LARGE",
                message=f"Chunk count {len(chunks)} exceeds maximum limit of {self.config.max_chunks_per_book}."
            )

        # 6. Atomic Cleanup if Overwriting
        if existing_book and overwrite:
            try:
                self.vector_store.delete_by_book(existing_book.book_id)
                self.workspace_mgr.remove_book(workspace_id, existing_book.book_id)
            except Exception as e:
                return IngestionResult(
                    status=IngestionStatus.FAILED,
                    error_code="ERR_VECTOR_STORAGE_FAILED",
                    message=f"Failed to clear old records during overwrite: {str(e)}"
                )

        # 7. Embed Chunks (Patient Retry Profile)
        chunk_texts = [c.text_content for c in chunks]
        try:
            vectors = await self.embedding_client.embed_texts(chunk_texts)
        except Exception as e:
            return IngestionResult(
                status=IngestionStatus.FAILED,
                error_code="ERR_UPSTREAM_EMBEDDING_FAILED",
                message=f"Failed to embed chunks after retry: {str(e)}"
            )

        # 8. Index into Scoped Vector Store
        try:
            self.vector_store.add_chunks(chunks, vectors)
        except Exception as e:
            return IngestionResult(
                status=IngestionStatus.FAILED,
                error_code="ERR_VECTOR_STORAGE_FAILED",
                message=f"Failed to persist vector records: {str(e)}"
            )

        # 9. Register Book Metadata
        book_meta = BookMetadata(
            book_id=book_id,
            workspace_id=workspace_id,
            title=book_title,
            total_pages=len(pages),
            total_chunks=len(chunks)
        )
        self.workspace_mgr.assign_book(workspace_id, book_meta)

        duration = time.time() - start_time
        total_tokens_est = sum(c.token_count for c in chunks)

        return IngestionResult(
            status=IngestionStatus.COMPLETED,
            book_id=book_id,
            book_title=book_title,
            total_pages=len(pages),
            total_chunks_indexed=len(chunks),
            estimated_total_tokens=total_tokens_est,
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

        # 1. Validate Workspace
        if not self.workspace_mgr.get_workspace(workspace_id):
            return QueryResult(
                query=query_text,
                workspace_id=workspace_id,
                status=QueryStatus.FAILED,
                error_code="ERR_WORKSPACE_NOT_FOUND",
                error_message=f"Workspace '{workspace_id}' does not exist."
            )

        # 2. Fast Query Embedding
        retrieval_start = time.time()
        try:
            query_vec = await self.embedding_client.embed_query(query_text)
        except Exception as e:
            return QueryResult(
                query=query_text,
                workspace_id=workspace_id,
                status=QueryStatus.FAILED,
                error_code="ERR_UPSTREAM_EMBEDDING_FAILED",
                error_message=f"Failed to embed query text: {str(e)}",
                total_latency_ms=round((time.time() - start_time) * 1000, 2)
            )

        # 3. Scoped Dense Retrieval
        try:
            retrieved_chunks = self.vector_store.search_scoped(
                workspace_id=workspace_id,
                query_vector=query_vec,
                top_k=k
            )
        except Exception as e:
            return QueryResult(
                query=query_text,
                workspace_id=workspace_id,
                status=QueryStatus.FAILED,
                error_code="ERR_VECTOR_STORAGE_FAILED",
                error_message=f"Vector store search failed: {str(e)}",
                total_latency_ms=round((time.time() - start_time) * 1000, 2)
            )

        retrieval_time_ms = (time.time() - retrieval_start) * 1000

        # Empty context handling
        if not retrieved_chunks:
            return QueryResult(
                query=query_text,
                workspace_id=workspace_id,
                status=QueryStatus.SUCCESS,
                answer="No relevant context found in this workspace.",
                citations=[],
                retrieval_time_ms=round(retrieval_time_ms, 2),
                generation_time_ms=0.0,
                total_latency_ms=round((time.time() - start_time) * 1000, 2)
            )

        # 4. Generation with Graceful Degradation
        gen_start = time.time()
        try:
            answer, citations, tokens = await self.generator.generate_response(
                query=query_text,
                retrieved_chunks=retrieved_chunks
            )
            gen_time_ms = (time.time() - gen_start) * 1000
            total_latency_ms = (time.time() - start_time) * 1000

            return QueryResult(
                query=query_text,
                workspace_id=workspace_id,
                status=QueryStatus.SUCCESS,
                answer=answer,
                citations=citations,
                retrieval_time_ms=round(retrieval_time_ms, 2),
                generation_time_ms=round(gen_time_ms, 2),
                total_latency_ms=round(total_latency_ms, 2),
                prompt_tokens=tokens.get("prompt_tokens", 0),
                completion_tokens=tokens.get("completion_tokens", 0),
                total_tokens=tokens.get("total_tokens", 0)
            )
        except Exception as e:
            gen_time_ms = (time.time() - gen_start) * 1000
            total_latency_ms = (time.time() - start_time) * 1000

            # Fallback: Partial grounding with retrieved citations
            fallback_citations = [
                Citation(
                    book_title=c.book_title,
                    page_start=c.page_start,
                    page_end=c.page_end,
                    relevance_score=round(score, 4),
                    text_content=c.text_content
                )
                for c, score in retrieved_chunks
            ]

            return QueryResult(
                query=query_text,
                workspace_id=workspace_id,
                status=QueryStatus.PARTIAL,
                answer="Dịch vụ tổng hợp câu trả lời tạm thời gián đoạn. Dưới đây là các đoạn trích liên quan nhất được tìm thấy từ tài liệu của bạn.",
                citations=fallback_citations,
                error_code="ERR_UPSTREAM_GENERATION_FAILED",
                error_message=f"LLM Generation service failed: {str(e)}",
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
git commit -m "feat(phase-3): add end-to-end EBookRAGPipeline with resilience, deduplication and graceful degradation"
```
