# 13F-INSIGHT

**A retrieval-augmented research assistant that answers questions about SEC 13F institutional holdings — grounded in the actual filings, not the model's memory.**

Ask "What was Berkshire Hathaway's largest position?" or "Which funds increased exposure to a given CUSIP?" and get an answer backed by specific accession numbers, share counts, and filing dates — every claim traceable to a source document.

---

## Why I built this

I wanted a research journal an analyst could actually trust. Off-the-shelf LLMs are confidently wrong about recent 13F filings: the data lands *after* their training cutoff, so they either refuse or hallucinate plausible-looking share counts. That failure mode is unacceptable for financial research, where a fabricated CUSIP or a rounded share count is worse than no answer at all.

13F-INSIGHT closes that gap. It ingests real quarterly 13F-HR filings from 100 institutional funds, indexes them for hybrid semantic + keyword retrieval, and forces the generator to answer **only** from retrieved filing rows — then audits every response against the source corpus. The result is a system that says "I don't have that" when it should, and cites exact figures when it can.

## Origin

Built as a post-graduate-level exercise in applied generative AI: design and evaluate a RAG system over a specialized, non-public dataset (SEC 13F institutional holdings) that falls outside any LLM's training data, comparing a baseline LLM against a retrieval-augmented pipeline on faithfulness and grounding. The 100-fund, multi-quarter corpus is deliberately chosen to sit beyond model knowledge cutoffs, so the baseline-vs-enhanced comparison measures what retrieval actually adds rather than what the model already memorized.

**Built with:** Python · LangChain · ChromaDB · BM25 (rank_bm25) · FlashRank (cross-encoder rerank) · Sentence-Transformers (all-MiniLM-L6-v2) · Gemini 3.1 Flash-Lite (generator) · GPT-4o-mini + Gemini 2.5 Pro (dual judge)

---

## See it in action

**Query:** *"Analyze the evolution of Nvidia (NVDA) holdings across the four reported quarters of 2025."*

**Baseline** — vanilla LLM, no retrieval:

> The 2025 calendar year has not yet concluded, and therefore four quarters of 2025 data do not exist. […] As of today, the most recent completed filing period is **Q3 2024**.

Confidently wrong. The model is reasoning from a stale world model — the filings exist, they're just past its training cutoff — so it invents "2024 trends" to fill the gap.

**13F-INSIGHT** — retrieval-augmented:

> Based on the provided 13F filing excerpts […] **BNY Mellon Investment Management** reported NVIDIA CORP (CUSIP: `67066G104`):
> • Period ending **2025-03-31** (Accession `0001390777-25-000071`): share counts 26,905; 4,362,490; … 75,774,092
> • Period ending **2025-06-30** (Accession `0001390777-25-000119`): … a marked decrease, with the largest Q1 position (75,774,092 shares) no longer present.
> No data was available for Q4 2025 across any of the provided excerpts.
>
> **Grounding: PASSED** — cited CUSIP `67066G104` verified against the source filings.

Real share counts and real accession numbers pulled from the actual filings — and where data is genuinely missing (Q4 2025), it *says so* instead of inventing it.

*Abridged from [`demo_log.txt`](demo_log.txt), which carries the full side-by-side with dual-judge scores and per-chunk retrieval provenance.*

---

## Architecture

```
SEC EDGAR → XML Parsing → Table-Aware Chunking → ChromaDB + BM25
         → Hybrid Retrieval (RRF) → FlashrankRerank → Stuff / Map-Reduce → Gemini 3.1 Flash-Lite
         → Dual-Judge Evaluation (GPT-4o-mini + Gemini 2.5 Pro) → Audit Trail
```

## Pipeline Stages

| Stage | Notebook | Purpose |
|-------|----------|---------|
| 1 | `Model/stage1_acquisition.ipynb` | Downloads up to 4 quarterly 13F-HR filings per fund for 100 CIKs from SEC EDGAR with rate-limit compliance |
| 2 | `Model/stage2_parsing.ipynb` | Extracts `<informationTable>` XML, converts to DataFrames, chunks into ≤300-word Markdown tables with metadata headers |
| 3 | `Model/stage3_vectordb.ipynb` | Builds ChromaDB vector store (all-MiniLM-L6-v2) and BM25 index; wires Hybrid Retriever with Reciprocal Rank Fusion |
| 4 | `Model/stage4_rag_analysis.ipynb` | Full RAG pipeline with entity/temporal/diversity filters, FlashrankRerank, Stuff/Map-Reduce chains, and audit trail logging |
| 5 | `Model/stage5_evaluation.ipynb` | Runs 10 diverse test queries (Baseline vs Enhanced), dual-judge scoring, grounding checks, persists `evaluation_results.json` |
| 6 | `Model/stage6_demo.ipynb` | 3 thematic demo queries with side-by-side output, generates `demo_log.txt` |

## Advanced Techniques

- **Hybrid Search** — BM25 (keyword) + ChromaDB (semantic) fused via Reciprocal Rank Fusion
- **Cross-Encoder Reranking** — FlashrankRerank with dual-width (narrow=8 / wide=20)
- **Map-Reduce Synthesis** — Parallel chunk extraction + consolidated reduction for cross-fund queries
- **Entity / Temporal / Diversity Filters** — Post-retrieval precision filters
- **Dual-Judge Evaluation** — GPT-4o-mini (OpenAI) + Gemini 2.5 Pro (Google) — cross-vendor, no self-grading
- **CUSIP Grounding Check** — 5MB corpus scan verifying cited identifiers against source filings

---

## Results

Ten diverse queries (fact lookup, numerical extraction, comparative synthesis, CUSIP lookup, hallucination traps, out-of-scope periods, temporal comparisons) were scored 1–5 by two independent LLM judges from different vendors. The retrieval-augmented pipeline more than doubled faithfulness over the vanilla baseline:

| Metric | Value |
|--------|-------|
| Baseline avg faithfulness | **1.75 / 5** |
| Enhanced avg faithfulness | **4.45 / 5** |
| Overall improvement | **+2.70 points** |
| Grounding pass rate | 7/10 (2 indeterminate on justified refusal, 1 known false-positive on Q4 — see Design Notes) |

### Per-query breakdown

| Query | Type | Baseline | Enhanced | Δ | Grounded |
|-------|------|:--------:|:--------:|:--:|:--------:|
| Q01 | Fact-based | 1.5 | 5.0 | +3.5 | Yes |
| Q02 | Numerical Extraction | 1.0 | 4.5 | +3.5 | Yes |
| Q03 | Comparative / Deep Context | 1.5 | 4.5 | +3.0 | Yes |
| Q04 | Sector Concentration | 1.5 | 3.0 | +1.5 | No |
| Q05 | CUSIP Lookup (BM25) | 1.0 | 5.0 | +4.0 | Yes |
| Q06 | Hallucination Test | 3.0 | 5.0 | +2.0 | N/A |
| Q07 | Out-of-Scope Period | 3.0 | 5.0 | +2.0 | N/A |
| Q08 | Filing Date Precision | 1.0 | 5.0 | +4.0 | Yes |
| Q09 | Cross-Period Comparison | 3.0 | 3.0 | +0.0 | Yes |
| Q10 | Specific Fund Period | 1.5 | 4.5 | +3.0 | N/A |
| **Avg** | | **1.75** | **4.45** | **+2.70** | |

The two queries where the baseline was competitive (Q06, Q07) are exactly the ones designed to be *unanswerable* — a non-existent entity and an out-of-scope period. There, a good system should refuse; the enhanced pipeline refuses correctly and gets full marks for it. The full reasoning behind each score, and the honest failure modes (Q04's grounding false-positive, Q09's flat cross-period result), are written up in [Design Notes & Evaluation](DESIGN_NOTES.md).

---

## Quickstart

> **First run builds the corpus from scratch (~30–45 min).** There's no committed data — Stages 1–3 download the filings from SEC EDGAR and build the index (Stage 1 alone is ~15–30 min against SEC's rate limit). Once that's done, Stages 4–6 run locally against the built corpus in minutes.

### Prerequisites

- Python 3.10+
- Jupyter Notebook / VS Code with the Jupyter extension

### API keys

Copy the two template files in `Model/`, drop your keys in, and rename off the `.example` suffix (one key per file, no quotes):

```bash
cp Model/gemini_api_key.example.txt  Model/gemini_api_key.txt
cp Model/github_token.example.txt    Model/github_token.txt
# then edit each file and paste your key
```

- `Model/gemini_api_key.txt` — Google AI Studio API key (Gemini generator + judge). Get one at https://aistudio.google.com/apikey
- `Model/github_token.txt` — GitHub personal access token (GPT-4o-mini evaluator via GitHub Models). Create one at https://github.com/settings/tokens

Both real key files are gitignored, so they will never be committed.

### Install dependencies

Each notebook installs its own dependencies in the first code cell. To install everything up front:

```bash
pip install sec-edgar-downloader tqdm beautifulsoup4 lxml tabulate pandas \
            langchain langchain-community langchain-huggingface langchain-chroma \
            sentence-transformers chromadb rank_bm25 flashrank openai google-generativeai
```

### Run

Run the notebooks **sequentially** — each stage consumes the previous stage's output:

```
Stage 1 → Stage 2 → Stage 3 → Stage 4 → Stage 5 → Stage 6
```

1. **Stage 1** — ~15–30 min (SEC rate limits). Creates `Model/data/13f_filings/`.
2. **Stage 2** — ~2 min. Creates `Model/data/processed_holdings.json`.
3. **Stage 3** — ~5–10 min (embedding). Creates `Model/vector_db/local_chroma_storage/`.
4. **Stage 4** — ~5 min. Creates `Model/system_audit_trail.json`.
5. **Stage 5** — ~15–20 min (API calls). Creates `Model/data/evaluation_results.json`.
6. **Stage 6** — ~10 min. Creates `demo_log.txt`.

## Regenerating the data

The `Model/data/` corpus and `Model/vector_db/` index are **not** checked into the repo (they're large and fully reproducible). To rebuild them from scratch:

- **`Model/data/`** — run **Stage 1** (downloads raw 13F-HR filings from SEC EDGAR) followed by **Stage 2** (parses and chunks them into `processed_holdings.json`).
- **`Model/vector_db/`** — run **Stage 3**, which embeds the chunks into ChromaDB and builds the BM25 index.

Stage 1 hits the SEC EDGAR fair-access endpoint, so it requires network access and takes the longest; everything downstream runs locally off its output.

## Key Outputs

| File | Description |
|------|-------------|
| `demo_log.txt` | Sample inputs, system outputs, and technical commentary |
| `Model/system_audit_trail.json` | Per-query audit trail with accession numbers, retrieved chunks, scores |
| `Model/data/evaluation_results.json` | Full evaluation: 10 queries, baseline vs enhanced, dual-judge scores |
| `Model/data/processed_holdings.json` | Parsed and chunked 13F holdings data |

## Design Notes

The engineering reflection — how the pipeline evolved across 16 improvements, the retrieval-precision gap, the unaudited map-step "hallucination laundering" risk, and every honest failure mode — lives in [DESIGN_NOTES.md](DESIGN_NOTES.md).

## License & disclaimer

Released under the [MIT License](LICENSE).

Educational / portfolio project. 13F-HR filings are public-domain data published by the U.S. Securities and Exchange Commission via [EDGAR](https://www.sec.gov/edgar); this project is not affiliated with, endorsed by, or sponsored by the SEC or any fund named in the dataset. Nothing here is investment advice. When pulling from SEC EDGAR, set a real contact email in the Stage 1 User-Agent to comply with SEC's [fair-access policy](https://www.sec.gov/os/webmaster-faq#developers).
