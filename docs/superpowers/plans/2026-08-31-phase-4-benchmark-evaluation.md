# Phase 4: Benchmark & Evaluation Implementation Plan (Managed with uv)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Xây dựng Phase 4 thiết lập hệ thống kiểm thử tự động (Benchmark Engine) đo lường Recall@5, Faithfulness và Latency trên tập dữ liệu Golden Test Set, xuất bản Baseline Performance Report. Quản lý thực thi qua `uv`.

**Architecture:** Nhận `EBookRAGPipeline` từ Phase 3 -> Tự động chạy tập câu hỏi đánh giá theo từng Workspace qua `uv run` -> Tính toán tỷ lệ Recall@5 trang trích dẫn và Faithfulness câu trả lời -> So sánh với ngưỡng chất lượng (Recall $\ge 70\%$, Faithfulness $\ge 85\%$) -> Xuất báo cáo hiệu năng.

**Tech Stack:** `uv`, Python 3.10+, `pydantic`, `pytest`, `pytest-asyncio`.

**Spec:** `docs/superpowers/specs/2026-08-31-ebook-rag-baseline-hld.md`

## Global Constraints

- **Environment & Execution**: Toàn bộ các lệnh chạy test, cài đặt dependencies và thực thi phải thông qua **`uv`** (ví dụ: `uv run pytest`).
- **Input Hand-off**: Nhận `EBookRAGPipeline` hoàn chỉnh từ Phase 3.
- **Evaluation Criteria**: Retrieval Recall@5 $\ge 70\%$, Answer Faithfulness $\ge 85\%$, Latency $\le 2.5$s.
- **Output Hand-off**: Xuất `BenchmarkReport` đóng vai trò là mốc so sánh (Baseline Benchmark) cho các phiên bản tiếp theo.

---

### Task 1: Benchmark Evaluation Engine with Recall & Faithfulness Metrics

**Files:**
- Create: `src/ebook_rag/evaluation/benchmark.py`
- Test: `tests/evaluation/test_benchmark.py`

**Interfaces:**
- Consumes: `EBookRAGPipeline`, `QueryResult`
- Produces: `TestCase`, `BenchmarkReport`, `BenchmarkRunner.run_evaluation()`

- [ ] **Step 1: Write failing unit test**

```python
# tests/evaluation/test_benchmark.py
import pytest
from unittest.mock import AsyncMock
from src.ebook_rag.evaluation.benchmark import BenchmarkRunner, TestCase
from src.ebook_rag.pipeline import EBookRAGPipeline
from src.ebook_rag.core.models import QueryResult, Citation

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
    assert report.faithfulness_rate == 1.0
    assert report.avg_latency_ms == 600.0
    assert report.passed_threshold is True
```

- [ ] **Step 2: Run test using uv to verify it fails**

Run: `uv run pytest tests/evaluation/test_benchmark.py -v`
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

            # Recall@K check based on citation page number
            if tc.expected_page is not None:
                retrieved_pages = [c.page_number for c in res.citations]
                if tc.expected_page in retrieved_pages:
                    correct_retrievals += 1
            else:
                correct_retrievals += 1

            # Faithfulness check based on keyword grounding
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

- [ ] **Step 4: Run test using uv to verify it passes**

Run: `uv run pytest tests/evaluation/test_benchmark.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ebook_rag/evaluation/ tests/evaluation/
git commit -m "feat(phase-4): add Phase 4 benchmark runner and evaluation metrics"
```
