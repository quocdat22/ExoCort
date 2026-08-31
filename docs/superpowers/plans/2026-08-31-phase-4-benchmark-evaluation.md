# Phase 4: Benchmark & Evaluation Implementation Plan (Managed with uv)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Xây dựng Phase 4 thiết lập hệ thống kiểm thử tự động (Benchmark Engine) đo lường Recall@K, Faithfulness và Latency trên tập dữ liệu Golden Test Set, xuất bản Baseline Performance Report. Quản lý thực thi qua `uv`.

**Architecture:** Nhận `EBookRAGPipeline` từ Phase 3 -> Tự động chạy tập câu hỏi đánh giá theo từng Workspace qua `uv run` -> Tính toán tỷ lệ Recall@K trang trích dẫn và Faithfulness câu trả lời (LLM-as-judge) -> So sánh với ngưỡng chất lượng (Recall $\ge 70\%$, Faithfulness $\ge 85\%$) -> Xuất báo cáo hiệu năng.

**Tech Stack:** `uv`, Python 3.10+, `pydantic`, `pytest`, `pytest-asyncio`.

**Spec:** `docs/superpowers/specs/2026-08-31-exocort-baseline-hld.md`

## Global Constraints

- **Environment & Execution**: Toàn bộ các lệnh chạy test, cài đặt dependencies và thực thi phải thông qua **`uv`** (ví dụ: `uv run pytest`).
- **Input Hand-off**: Nhận `EBookRAGPipeline` hoàn chỉnh từ Phase 3.
- **Evaluation Criteria**: Retrieval Recall@K $\ge 70\%$, Answer Faithfulness $\ge 85\%$, Latency $\le 2.5$s.
- **Output Hand-off**: Xuất `BenchmarkReport` đóng vai trò là mốc so sánh (Baseline Benchmark) cho các phiên bản tiếp theo.

---

### Task 1: Golden Test Dataset & Loader

**Files:**
- Create: `eval/golden_test_set.json`
- Create: `src/exocort/evaluation/dataset.py`
- Test: `tests/evaluation/test_dataset.py`

**Interfaces:**
- Produces: `TestCase`, `load_test_cases()`

- [ ] **Step 1: Write failing unit test**

```python
# tests/evaluation/test_dataset.py
import pytest
import json
from exocort.evaluation.dataset import load_test_cases, TestCase

def test_load_test_cases(tmp_path):
    data = [
        {"query": "What is SRP?", "expected_page": 138, "expected_keywords": ["single responsibility"]},
        {"query": "What is DIP?", "expected_page": 150}
    ]
    path = tmp_path / "test_cases.json"
    path.write_text(json.dumps(data))
    
    cases = load_test_cases(path)
    assert len(cases) == 2
    assert cases[0].expected_page == 138
    assert cases[1].expected_keywords == []
```

- [ ] **Step 2: Run test using uv to verify it fails**

Run: `uv run pytest tests/evaluation/test_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/exocort/evaluation/dataset.py
import json
from pathlib import Path
from typing import List
from pydantic import BaseModel

class TestCase(BaseModel):
    query: str
    expected_page: int | None = None
    expected_keywords: List[str] = []

def load_test_cases(path: Path) -> List[TestCase]:
    with open(path) as f:
        data = json.load(f)
    return [TestCase(**item) for item in data]
```

- [ ] **Step 4: Run test using uv to verify it passes**

Run: `uv run pytest tests/evaluation/test_dataset.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/exocort/evaluation/ tests/evaluation/
git commit -m "feat(phase-4): add golden test dataset loader"
```

---

### Task 2: Evaluation Metrics (Recall@K + LLM-as-Judge Faithfulness)

**Files:**
- Create: `src/exocort/evaluation/metrics.py`
- Test: `tests/evaluation/test_metrics.py`

**Interfaces:**
- Consumes: `Citation`
- Produces: `recall_at_k()`, `faithfulness_llm_judge()`

- [ ] **Step 1: Write failing unit test**

```python
# tests/evaluation/test_metrics.py
import pytest
from exocort.evaluation.metrics import recall_at_k, keyword_coverage, faithfulness_llm_judge
from exocort.core.models import Citation

def test_recall_at_k_hit():
    citations = [
        Citation(book_title="Book A", page_start=135, page_end=140, relevance_score=0.9, text_content="SRP excerpt")
    ]
    assert recall_at_k(citations, expected_page=138) is True

def test_recall_at_k_miss():
    citations = [
        Citation(book_title="Book A", page_start=10, page_end=12, relevance_score=0.8, text_content="Intro excerpt")
    ]
    assert recall_at_k(citations, expected_page=138) is False

def test_recall_at_k_no_ground_truth():
    assert recall_at_k([], expected_page=None) is True

def test_keyword_coverage():
    citations = [
        Citation(book_title="Clean Architecture", page_start=135, page_end=140, text_content="Single responsibility principle states a module has one reason to change.")
    ]
    assert keyword_coverage(citations, ["single responsibility", "reason to change"]) == 1.0
    assert keyword_coverage(citations, ["single responsibility", "database schema"]) == 0.5
    assert keyword_coverage(citations, []) == 1.0

@pytest.mark.asyncio
async def test_faithfulness_llm_judge_exact():
    async def mock_judge(prompt: str) -> str:
        return "0.85"

    score = await faithfulness_llm_judge(
        question="What is SRP?",
        answer="SRP says a class should have one reason to change.",
        source_texts=["The Single Responsibility Principle states that a class should have one reason to change."],
        judge_fn=mock_judge,
    )
    assert score == 0.85

@pytest.mark.asyncio
async def test_faithfulness_llm_judge_with_explanation_and_formatting():
    # LLM returns text with commentary
    async def mock_judge_verbose(prompt: str) -> str:
        return "0.95 - The answer is supported by excerpt 1."

    score1 = await faithfulness_llm_judge("q", "a", ["s"], judge_fn=mock_judge_verbose)
    assert score1 == 0.95

    # LLM returns markdown bold score
    async def mock_judge_markdown(prompt: str) -> str:
        return "**1.0**\n\nExplanation: Completely grounded."

    score2 = await faithfulness_llm_judge("q", "a", ["s"], judge_fn=mock_judge_markdown)
    assert score2 == 1.0

@pytest.mark.asyncio
async def test_faithfulness_llm_judge_invalid_response():
    async def mock_judge(prompt: str) -> str:
        return "not a number without any digits"

    score = await faithfulness_llm_judge(
        question="q", answer="a", source_texts=["s"],
        judge_fn=mock_judge,
    )
    assert score == 0.0
```

- [ ] **Step 2: Run test using uv to verify it fails**

Run: `uv run pytest tests/evaluation/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/exocort/evaluation/metrics.py
import re
from typing import List, Callable, Awaitable, Optional
from exocort.core.models import Citation

def recall_at_k(
    citations: List[Citation],
    expected_page: Optional[int] = None,
    expected_keywords: Optional[List[str]] = None,
) -> bool:
    """Check if expected_page matches citation ranges or expected_keywords are found."""
    if expected_page is not None:
        return any(c.page_start <= expected_page <= c.page_end for c in citations)
    if expected_keywords:
        return keyword_coverage(citations, expected_keywords) > 0.0
    return True

def keyword_coverage(citations: List[Citation], expected_keywords: List[str]) -> float:
    """Calculate ratio of expected keywords found in retrieved citations."""
    if not expected_keywords:
        return 1.0
    if not citations:
        return 0.0
    combined_text = " ".join((c.text_content or "") for c in citations).lower()
    matches = sum(1 for kw in expected_keywords if kw.strip().lower() in combined_text)
    return matches / len(expected_keywords)

FAITHFULNESS_PROMPT = """You are an impartial evaluator. Given a question, an answer, and source excerpts, 
rate the **faithfulness** of the answer on a scale from 0.0 to 1.0.

Faithfulness means: every claim in the answer is directly supported by the source excerpts.
- 1.0 = all claims grounded in sources
- 0.0 = answer contains fabricated information not in sources

## Question
{question}

## Answer
{answer}

## Source Excerpts
{sources}

Respond with ONLY a decimal number between 0.0 and 1.0."""

async def faithfulness_llm_judge(
    question: str,
    answer: str,
    source_texts: List[str],
    judge_fn: Callable[[str], Awaitable[str]],
) -> float:
    """Use LLM-as-judge to evaluate answer faithfulness with robust regex fallback."""
    if not source_texts or not answer.strip():
        return 0.0
        
    sources_str = "\n---\n".join(
        f"Excerpt {i+1}: {text}" for i, text in enumerate(source_texts)
    )
    prompt = FAITHFULNESS_PROMPT.format(
        question=question, answer=answer, sources=sources_str
    )
    score_str = await judge_fn(prompt)
    cleaned = score_str.strip()

    # 1. Direct float attempt
    try:
        return max(0.0, min(1.0, float(cleaned)))
    except ValueError:
        pass

    # 2. Extract first valid float/integer bounded between 0.0 and 1.0
    match = re.search(r"\b(1(?:\.0+)?|0(?:\.\d+)?)\b", cleaned)
    if match:
        try:
            return max(0.0, min(1.0, float(match.group(1))))
        except ValueError:
            pass

    # 3. Looser extraction (e.g. bolded **0.95**)
    match = re.search(r"(1(?:\.0+)?|0(?:\.\d+)?)", cleaned)
    if match:
        try:
            return max(0.0, min(1.0, float(match.group(1))))
        except ValueError:
            pass

    return 0.0
```

- [ ] **Step 4: Run test using uv to verify it passes**

Run: `uv run pytest tests/evaluation/test_metrics.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/exocort/evaluation/ tests/evaluation/
git commit -m "feat(phase-4): add evaluation metrics for recall and faithfulness"
```

---

### Task 3: BenchmarkRunner & Report

**Files:**
- Create: `src/exocort/evaluation/benchmark.py`
- Test: `tests/evaluation/test_benchmark.py`

**Interfaces:**
- Consumes: `EBookRAGPipeline`, `QueryResult`, `TestCase`
- Produces: `BenchmarkReport`, `BenchmarkRunner.run_evaluation()`

- [ ] **Step 1: Write failing unit test**

```python
# tests/evaluation/test_benchmark.py
import pytest
from unittest.mock import AsyncMock
from pydantic import SecretStr
from exocort.evaluation.benchmark import BenchmarkRunner, BenchmarkReport
from exocort.evaluation.dataset import TestCase
from exocort.pipeline import EBookRAGPipeline
from exocort.core.config import RAGConfig
from exocort.core.models import QueryResult, Citation

@pytest.mark.asyncio
async def test_benchmark_runner_with_judge():
    config = RAGConfig(
        jina_api_key=SecretStr("mock_jina_key"),
        openrouter_api_key=SecretStr("mock_openrouter_key"),
    )
    pipeline = EBookRAGPipeline(config)
    pipeline.query_workspace = AsyncMock(return_value=QueryResult(
        query="What is SRP?",
        workspace_id="ws_1",
        answer="SRP states a class should have one reason to change.",
        citations=[Citation(
            book_title="Clean Code",
            page_start=138,
            page_end=140,
            relevance_score=0.9,
            text_content="A class should have one, and only one, reason to change.",
        )],
        retrieval_time_ms=100.0,
        generation_time_ms=500.0,
        total_latency_ms=600.0,
    ))

    async def mock_judge(prompt: str) -> str:
        assert "A class should have one, and only one, reason to change." in prompt
        return "0.95"

    test_cases = [
        TestCase(query="What is SRP?", expected_page=138, expected_keywords=["reason to change"])
    ]

    runner = BenchmarkRunner(pipeline, judge_fn=mock_judge, concurrency_limit=2)
    report = await runner.run_evaluation("ws_1", test_cases)

    assert report.total_test_cases == 1
    assert report.recall_at_k == 1.0
    assert report.keyword_coverage_rate == 1.0
    assert report.faithfulness_rate == 0.95
    assert report.avg_latency_ms == 600.0
    assert report.passed_threshold is True

@pytest.mark.asyncio
async def test_benchmark_runner_concurrent_execution():
    config = RAGConfig(
        jina_api_key=SecretStr("mock_jina_key"),
        openrouter_api_key=SecretStr("mock_openrouter_key"),
    )
    pipeline = EBookRAGPipeline(config)

    active_queries = 0
    max_active_queries = 0

    async def mock_query(workspace_id: str, query: str):
        nonlocal active_queries, max_active_queries
        active_queries += 1
        max_active_queries = max(max_active_queries, active_queries)
        import asyncio
        await asyncio.sleep(0.02)
        active_queries -= 1
        return QueryResult(
            query=query,
            workspace_id=workspace_id,
            answer=f"Answer for {query}",
            citations=[Citation(
                book_title="Book",
                page_start=10,
                page_end=12,
                relevance_score=0.9,
                text_content="key excerpt",
            )],
            retrieval_time_ms=50.0,
            generation_time_ms=100.0,
            total_latency_ms=150.0,
            total_tokens=50,
        )

    pipeline.query_workspace = mock_query

    test_cases = [
        TestCase(query=f"Query {i}", expected_page=10, expected_keywords=["key excerpt"])
        for i in range(4)
    ]

    runner = BenchmarkRunner(pipeline, concurrency_limit=2)
    report = await runner.run_evaluation("ws_1", test_cases)

    assert report.total_test_cases == 4
    assert report.recall_at_k == 1.0
    assert report.passed_threshold is True
    assert max_active_queries == 2
```

- [ ] **Step 2: Run test using uv to verify it fails**

Run: `uv run pytest tests/evaluation/test_benchmark.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/exocort/evaluation/benchmark.py
import asyncio
import statistics
from typing import List, Optional, Callable, Awaitable
from pydantic import BaseModel
from exocort.pipeline import EBookRAGPipeline
from exocort.evaluation.dataset import TestCase
from exocort.evaluation.metrics import recall_at_k, keyword_coverage, faithfulness_llm_judge

class BenchmarkReport(BaseModel):
    total_test_cases: int
    recall_at_k: float
    keyword_coverage_rate: float = 1.0
    faithfulness_rate: float
    avg_latency_ms: float
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    total_tokens_consumed: int = 0
    passed_threshold: bool

class BenchmarkRunner:
    def __init__(
        self,
        pipeline: EBookRAGPipeline,
        judge_fn: Optional[Callable[[str], Awaitable[str]]] = None,
        concurrency_limit: int = 5,
    ):
        self.pipeline = pipeline
        self.judge_fn = judge_fn
        self.concurrency_limit = concurrency_limit

    async def run_evaluation(
        self,
        workspace_id: str,
        test_cases: List[TestCase],
        concurrency_limit: Optional[int] = None,
    ) -> BenchmarkReport:
        if not test_cases:
            return BenchmarkReport(
                total_test_cases=0, recall_at_k=0.0,
                keyword_coverage_rate=0.0,
                faithfulness_rate=0.0, avg_latency_ms=0.0,
                p95_latency_ms=0.0, p99_latency_ms=0.0,
                passed_threshold=False,
            )

        limit = concurrency_limit or self.concurrency_limit
        sem = asyncio.Semaphore(limit)

        async def _eval_case(tc: TestCase):
            async with sem:
                res = await self.pipeline.query_workspace(workspace_id, tc.query)
                lat = res.total_latency_ms
                tokens = res.total_tokens

                # Recall@K
                is_recalled = 1.0 if recall_at_k(res.citations, tc.expected_page, tc.expected_keywords) else 0.0

                # Keyword Coverage Metric
                if tc.expected_keywords:
                    kw_cov = keyword_coverage(res.citations, tc.expected_keywords)
                else:
                    kw_cov = 1.0

                # Faithfulness via LLM-as-judge
                if self.judge_fn:
                    source_texts = [c.text_content for c in res.citations if c.text_content]
                    faith_score = await faithfulness_llm_judge(
                        question=tc.query,
                        answer=res.answer,
                        source_texts=source_texts,
                        judge_fn=self.judge_fn,
                    )
                else:
                    faith_score = 1.0

                return lat, tokens, is_recalled, kw_cov, faith_score

        eval_results = await asyncio.gather(*[_eval_case(tc) for tc in test_cases])

        latencies = [r[0] for r in eval_results]
        total_tokens = sum(r[1] for r in eval_results)
        correct_retrievals = sum(r[2] for r in eval_results)
        total_kw_coverage = sum(r[3] for r in eval_results)
        total_faithfulness = sum(r[4] for r in eval_results)

        n = len(test_cases)
        recall = correct_retrievals / n
        kw_cov = total_kw_coverage / n
        faithfulness = total_faithfulness / n
        avg_lat = sum(latencies) / n
        sorted_latencies = sorted(latencies)
        p95_idx = min(int(n * 0.95), n - 1)
        p99_idx = min(int(n * 0.99), n - 1)
        p95_lat = sorted_latencies[p95_idx]
        p99_lat = sorted_latencies[p99_idx]

        passed = (recall >= 0.70) and (faithfulness >= 0.85) and (p95_lat <= 2500.0)

        return BenchmarkReport(
            total_test_cases=n,
            recall_at_k=round(recall, 4),
            keyword_coverage_rate=round(kw_cov, 4),
            faithfulness_rate=round(faithfulness, 4),
            avg_latency_ms=round(avg_lat, 2),
            p95_latency_ms=round(p95_lat, 2),
            p99_latency_ms=round(p99_lat, 2),
            total_tokens_consumed=total_tokens,
            passed_threshold=passed,
        )
```

- [ ] **Step 4: Run test using uv to verify it passes**

Run: `uv run pytest tests/evaluation/test_benchmark.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/exocort/evaluation/ tests/evaluation/
git commit -m "feat(phase-4): add Phase 4 benchmark runner and evaluation report"
```

---

## Test Strategy

| Task | Component | Test Type | Goal |
|---|---|---|---|
| Task 1 | Golden Test Dataset & Loader | Unit Test | Verify `load_test_cases` parses JSON correctly into `TestCase` instances with proper expected page mapping. |
| Task 2 | Evaluation Metrics | Unit Test | Verify `recall_at_k` works with page ranges and `faithfulness_llm_judge` correctly parses LLM response. |
| Task 3 | BenchmarkRunner & Report | Unit Test | Verify the runner accurately aggregates metrics (Recall@K, Faithfulness) and reports pass/fail thresholds. |

## Definition of Done (DoD)

- [ ] All 3 tasks have 100% passing automated tests.
- [ ] Golden test dataset loads successfully.
- [ ] Evaluation metrics handle page range citations correctly (`page_start`, `page_end`).
- [ ] Faithfulness is evaluated using an LLM-as-judge approach instead of keyword matching.
- [ ] Benchmark runner successfully orchestrates execution of evaluation and asserts baseline thresholds (Recall $\ge 70\%$, Faithfulness $\ge 85\%$).
- [ ] Báo cáo (Benchmark Report) được xuất định dạng chính xác.
- [ ] Package rename to `exocort` is strictly enforced.

## Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| LLM-as-judge is slow or costly | Medium | Batch evaluation or use a smaller/cheaper model for evaluation. |
| Parsing LLM judge response fails | Low | Fallback to `0.0` as implemented, ensuring failures don't crash the pipeline. |
| Empty citations for faithfulness | Low | Handled gracefully when citations list or `text_content` is empty (LLM judge receives empty sources, properly scoring ungrounded responses as 0.0). |
