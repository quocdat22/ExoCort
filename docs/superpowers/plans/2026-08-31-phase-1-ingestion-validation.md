# Phase 1: Ingestion & Validation Implementation Plan (Managed with uv)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Khởi tạo project với `uv`, xây dựng Domain Models, Workspace Manager, cơ chế tiếp nhận PDF, kiểm tra text layer (chặn PDF scan `ERR_UNSUPPORTED_SCANNED_PDF`) và phân mảnh cửa sổ cố định Flat-Window liên trang dựa trên `tiktoken`.

**Architecture:** Môi trường quản lý bởi `uv` -> Tiếp nhận file PDF nhị phân -> Trích xuất trang -> Đo mật độ ký tự hợp lệ để phát hiện PDF scan -> Nối toàn bộ văn bản các trang -> Phân mảnh cửa sổ trượt token-based (tiktoken `cl100k_base`) liên trang với ánh xạ trang ngược bằng character offset -> Gắn metadata (`workspace_id`, `book_title`, `page_start`, `page_end`).

**Tech Stack:** `uv`, Python 3.10+, `pydantic`, `pypdf`, `tiktoken`, `pytest`.

**Spec:** `docs/superpowers/specs/2026-08-31-ebook-rag-baseline-hld.md`

## Global Constraints

- **Environment & Execution**: Toàn bộ các lệnh chạy test, cài đặt dependencies và thực thi phải thông qua **`uv`** (ví dụ: `uv run pytest`).
- **Document Format**: Digital PDF tiếng Anh có text layer. Từ chối PDF scan với mã lỗi `ERR_UNSUPPORTED_SCANNED_PDF`.
- **Chunking**: Cửa sổ trượt cố định (Flat-Window) liên trang, kích thước 512 tokens, gối đầu 50 tokens (~10%). Đếm token bằng `tiktoken` (`cl100k_base`).
- **Output Hand-off**: Xuất ra danh sách `List[Chunk]` (Normalized Chunks Collection) làm đầu vào cho Phase 2.

---

### Task 0: Project & Virtual Environment Initialization with uv

**Files:**
- Create: `pyproject.toml`
- Create: `src/exocort/__init__.py`
- Create: `src/exocort/core/__init__.py`
- Create: `src/exocort/ingestion/__init__.py`
- Create: `src/exocort/workspace/__init__.py`
- Test: `uv run python --version`

- [ ] **Step 1: Initialize project using uv**

```bash
uv init --lib --name exocort
uv add pydantic pypdf tiktoken
uv add --dev pytest pytest-asyncio
```

- [ ] **Step 2: Create `__init__.py` for all sub-packages**

```bash
mkdir -p src/exocort/core src/exocort/ingestion src/exocort/workspace
touch src/exocort/__init__.py
touch src/exocort/core/__init__.py
touch src/exocort/ingestion/__init__.py
touch src/exocort/workspace/__init__.py
```

- [ ] **Step 3: Verify environment with uv**

Run: `uv run pytest --version`
Expected: pytest version output

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock src/
git commit -m "chore(phase-1): initialize project structure and dependencies with uv"
```

---

### Task 1: Core Domain Models & RAG Configuration

**Files:**
- Create: `src/exocort/core/models.py`
- Create: `src/exocort/core/config.py`
- Test: `tests/core/test_models.py`

**Interfaces:**
- Produces: `Workspace`, `BookMetadata`, `Chunk`, `Citation`, `QueryResult`, `IngestionResult`, `IngestionStatus`, `RAGConfig`

- [ ] **Step 1: Write failing unit test**

```python
# tests/core/test_models.py
import pytest
from exocort.core.models import (
    Workspace,
    BookMetadata,
    Chunk,
    Citation,
    QueryResult,
    IngestionResult,
    IngestionStatus,
)
from exocort.core.config import RAGConfig

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
        page_start=15,
        page_end=16,
        text_content="Functions should do one thing.",
        chunk_index=0
    )
    assert chunk.page_start == 15
    assert chunk.page_end == 16
    assert chunk.workspace_id == "ws_01"

def test_ingestion_result_rejected_scan():
    res = IngestionResult(
        status=IngestionStatus.REJECTED,
        error_code="ERR_UNSUPPORTED_SCANNED_PDF",
        message="Scanned PDF format without valid text layer is not supported in MVP baseline."
    )
    assert res.status == IngestionStatus.REJECTED
    assert res.error_code == "ERR_UNSUPPORTED_SCANNED_PDF"

def test_rag_config_defaults():
    config = RAGConfig()
    assert config.chunk_size == 512
    assert config.chunk_overlap == 50
    assert config.min_valid_page_ratio == 0.5
    assert config.tokenizer_encoding == "cl100k_base"
```

- [ ] **Step 2: Run test using uv to verify it fails**

Run: `uv run pytest tests/core/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/exocort/core/config.py
from pydantic import BaseModel

class RAGConfig(BaseModel):
    chunk_size: int = 512
    chunk_overlap: int = 50
    min_text_density_per_page: int = 50
    min_valid_page_ratio: float = 0.5
    tokenizer_encoding: str = "cl100k_base"
    embedding_model: str = "jina-embeddings-v5-omni-small"
    embedding_batch_size: int = 32
    llm_model: str = "deepseek/deepseek-v4-flash-0731"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024
    top_k: int = 5
    jina_api_key: str = ""
    openrouter_api_key: str = ""
    chroma_persist_dir: str = "./chroma_data"
```

```python
# src/exocort/core/models.py
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
    page_start: int
    page_end: int
    text_content: str
    token_count: int = 0
    chunk_index: int

class Citation(BaseModel):
    book_title: str
    page_start: int
    page_end: int
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

- [ ] **Step 4: Run test using uv to verify it passes**

Run: `uv run pytest tests/core/test_models.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/exocort/core/ tests/core/
git commit -m "feat(phase-1): add core domain models and configuration"
```

---

### Task 2: Workspace Management Module

**Files:**
- Create: `src/exocort/workspace/manager.py`
- Test: `tests/workspace/test_manager.py`

**Interfaces:**
- Consumes: `Workspace`
- Produces: `WorkspaceManager.create_workspace()`, `WorkspaceManager.get_workspace()`, `WorkspaceManager.list_workspaces()`, `WorkspaceManager.assign_book()`, `WorkspaceManager.list_workspace_books()`

- [ ] **Step 1: Write failing unit test**

```python
# tests/workspace/test_manager.py
import pytest
from exocort.workspace.manager import WorkspaceManager

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

- [ ] **Step 2: Run test using uv to verify it fails**

Run: `uv run pytest tests/workspace/test_manager.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/exocort/workspace/manager.py
import uuid
from typing import Dict, List, Optional, Set
from exocort.core.models import Workspace

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

- [ ] **Step 4: Run test using uv to verify it passes**

Run: `uv run pytest tests/workspace/test_manager.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/exocort/workspace/ tests/workspace/
git commit -m "feat(phase-1): add workspace manager module"
```

---

### Task 3: PDF Extractor, Scanned PDF Validator & Flat-Window Chunker

**Files:**
- Create: `src/exocort/ingestion/extractor.py`
- Create: `src/exocort/ingestion/validator.py`
- Create: `src/exocort/ingestion/chunker.py`
- Test: `tests/ingestion/test_ingestion_pipeline.py`

**Interfaces:**
- Consumes: `RAGConfig`, `Chunk`
- Produces: `PDFExtractor.extract_pages()`, `PDFValidator.validate_text_layer()`, `FlatWindowChunker.chunk_pages()`

- [ ] **Step 1: Write failing unit test**

```python
# tests/ingestion/test_ingestion_pipeline.py
import pytest
from exocort.ingestion.validator import PDFValidator
from exocort.ingestion.chunker import FlatWindowChunker
from exocort.core.config import RAGConfig

def test_validator_rejects_empty_or_scanned_pdf():
    config = RAGConfig(min_text_density_per_page=50)
    validator = PDFValidator(config)

    # Scanned PDF: Pages contain almost no text
    pages_text = ["", "   ", "\n\n"]
    is_valid, error = validator.validate_text_layer(pages_text)
    assert is_valid is False
    assert error == "ERR_UNSUPPORTED_SCANNED_PDF"

def test_validator_accepts_valid_digital_pdf():
    config = RAGConfig(min_text_density_per_page=50)
    validator = PDFValidator(config)

    pages_text = [
        "This is a digital textbook chapter on distributed systems with plenty of valid text characters.",
        "Second page of the book discussing consensus algorithms such as Raft and Paxos."
    ]
    is_valid, error = validator.validate_text_layer(pages_text)
    assert is_valid is True
    assert error is None

def test_flat_window_chunker_cross_page():
    config = RAGConfig(chunk_size=20, chunk_overlap=5, tokenizer_encoding="cl100k_base")
    chunker = FlatWindowChunker(config)

    pages = [
        (1, "Functions should do one thing. They should do it well. They should do it only."),
        (2, "Clean code reads like well-written prose. It is crisp and clear.")
    ]
    chunks = chunker.chunk_pages(
        pages=pages,
        workspace_id="ws_01",
        book_id="bk_01",
        book_title="Clean Code"
    )

    assert len(chunks) >= 1
    assert chunks[0].book_id == "bk_01"
    assert chunks[0].workspace_id == "ws_01"
    assert chunks[0].page_start >= 1
    assert chunks[0].token_count <= 20
    # Verify cross-page: last chunk should span or start on page 2
    assert chunks[-1].page_end == 2

def test_flat_window_chunker_empty():
    config = RAGConfig(chunk_size=20, chunk_overlap=5)
    chunker = FlatWindowChunker(config)
    chunks = chunker.chunk_pages([], "ws_01", "bk_01", "Book")
    assert chunks == []

def test_flat_window_chunker_single_page():
    config = RAGConfig(chunk_size=10, chunk_overlap=2)
    chunker = FlatWindowChunker(config)
    pages = [(1, "Hello world this is a test sentence for chunking.")]
    chunks = chunker.chunk_pages(pages, "ws_01", "bk_01", "Test")
    assert len(chunks) >= 1
    for chunk in chunks:
        assert chunk.page_start == 1
        assert chunk.page_end == 1
        assert chunk.token_count <= 10
```

- [ ] **Step 2: Run test using uv to verify it fails**

Run: `uv run pytest tests/ingestion/test_ingestion_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/exocort/ingestion/extractor.py
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

```python
# src/exocort/ingestion/validator.py
from typing import List, Tuple, Optional
from exocort.core.config import RAGConfig

class PDFValidator:
    def __init__(self, config: RAGConfig):
        self.min_density = config.min_text_density_per_page
        self.min_valid_page_ratio = config.min_valid_page_ratio

    def validate_text_layer(self, pages_text: List[str]) -> Tuple[bool, Optional[str]]:
        if not pages_text:
            return False, "ERR_UNSUPPORTED_SCANNED_PDF"

        valid_pages_count = 0
        for text in pages_text:
            clean_text = "".join(text.split())
            if len(clean_text) >= self.min_density:
                valid_pages_count += 1

        if valid_pages_count / len(pages_text) < self.min_valid_page_ratio:
            return False, "ERR_UNSUPPORTED_SCANNED_PDF"

        return True, None
```

```python
# src/exocort/ingestion/chunker.py
import uuid
from typing import List, Tuple
import tiktoken
from exocort.core.models import Chunk
from exocort.core.config import RAGConfig

class FlatWindowChunker:
    def __init__(self, config: RAGConfig):
        self.chunk_size = config.chunk_size
        self.chunk_overlap = config.chunk_overlap
        self.enc = tiktoken.get_encoding(config.tokenizer_encoding)

    def chunk_pages(
        self,
        pages: List[Tuple[int, str]],
        workspace_id: str,
        book_id: str,
        book_title: str
    ) -> List[Chunk]:
        if not pages:
            return []

        # 1. Concatenate all pages, track page boundaries
        full_text = ""
        page_map: List[Tuple[int, int, int]] = []  # (page_num, char_start, char_end)
        for page_num, page_text in pages:
            char_start = len(full_text)
            full_text += page_text + "\n"
            char_end = len(full_text)
            page_map.append((page_num, char_start, char_end))

        # 2. Tokenize full text
        tokens = self.enc.encode(full_text)
        if not tokens:
            return []

        # 3. Sliding window over tokens
        chunks: List[Chunk] = []
        chunk_idx = 0
        start = 0
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.enc.decode(chunk_tokens)

            # 4. Map to page numbers via character offsets
            chunk_char_start = len(self.enc.decode(tokens[:start])) if start > 0 else 0
            chunk_char_end = chunk_char_start + len(chunk_text)
            page_start, page_end = self._find_page_range(
                chunk_char_start, chunk_char_end, page_map
            )

            chunks.append(Chunk(
                chunk_id=f"chk_{uuid.uuid4().hex[:10]}",
                book_id=book_id,
                book_title=book_title,
                workspace_id=workspace_id,
                page_start=page_start,
                page_end=page_end,
                text_content=chunk_text,
                token_count=len(chunk_tokens),
                chunk_index=chunk_idx,
            ))

            if end >= len(tokens):
                break
            start += max(1, self.chunk_size - self.chunk_overlap)
            chunk_idx += 1

        return chunks

    @staticmethod
    def _find_page_range(
        char_start: int, char_end: int,
        page_map: List[Tuple[int, int, int]]
    ) -> Tuple[int, int]:
        """Find the first and last page that overlap with [char_start, char_end)."""
        first_page = page_map[0][0]
        last_page = page_map[-1][0]
        for page_num, p_start, p_end in page_map:
            if p_start <= char_start < p_end:
                first_page = page_num
            if p_start < char_end <= p_end:
                last_page = page_num
        return first_page, last_page
```

- [ ] **Step 4: Run test using uv to verify it passes**

Run: `uv run pytest tests/ingestion/test_ingestion_pipeline.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/exocort/ingestion/ tests/ingestion/
git commit -m "feat(phase-1): add PDF extractor, validator and flat-window chunker"
```
