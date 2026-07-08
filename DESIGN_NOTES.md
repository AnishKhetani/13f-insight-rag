# Design Notes & Evaluation

This is the engineering write-up behind [13F-INSIGHT](README.md): the decisions that shaped the pipeline, the sixteen improvements it took to get retrieval-augmented answers to actually beat the baseline, the evaluation methodology, and — most usefully for anyone building something similar — the failure modes I couldn't fully close.

I've tried to be honest here rather than promotional. Several of these notes are about things that *don't* work perfectly, because those are the parts worth reading.

---

## 1. The problem: a "Private Data Gap" a vanilla LLM can't cross

The dataset is deliberately hostile to a stock LLM. It's ~100 institutional funds' quarterly 13F-HR filings, filed February 2026, covering periods through 2025-12-31 — all of it *after* any current model's training cutoff. That produces two distinct failure modes in the baseline:

- **Knowledge cutoff.** Asked about Q4 2025 holdings, the vanilla model insists the period "has not yet occurred" or that "we are currently in 2024." It's reasoning from a stale world model.
- **Private data gap.** Even setting the date aside, 13F holdings at this granularity (exact CUSIPs, share counts, per-fund positions) simply aren't in the training corpus. The baseline either refuses outright — *"it is currently impossible to provide data from February 2026 filings"* — or, worse, hallucinates confident numbers.

The whole system is built to bridge that gap by retrieval: put the real filing rows in front of the generator and force it to answer only from them. Every demo entry logs a "Private Data Gap" note quantifying how many chunks were retrieved to answer a question the baseline couldn't.

---

## 2. Data & methodology

### Sourcing (Stage 1)

Stage 1 pulls up to four quarterly 13F-HR filings per fund for 100 CIKs directly from SEC EDGAR, with fair-access throttling (1.5s per request, batched, exponential backoff on 403/429). A `VALID_YEARS = {"2024","2025","2026"}` gate rejects stale filings, and every CIK resolves to one of four states (`SUCCESS` / `STALE` / `MISSING_FORM` / `RATE_LIMITED`) so the acquisition report explains exactly why any fund was dropped. Accession numbers are preserved end-to-end — they become the provenance backbone later.

### Table-aware parsing (Stage 2)

13F data is tabular, and naive text chunking destroys it. `parse_information_table()` extracts the `<informationTable>` XML (handling namespace-prefixed variants like `ns1:infoTable`) and converts each filing to a DataFrame over `[nameOfIssuer, titleOfClass, cusip, value, sshPrnamt, investmentDiscretion]`. `chunk_markdown_table()` then emits Markdown tables that **keep the column header in every chunk**, so a row never loses its schema. `OVERLAP_ROWS = 3` means holdings near a chunk boundary appear whole in at least one chunk.

### Chunking sized to the embedder

`CHUNK_WORD_LIMIT = 300` words is chosen to fit `all-MiniLM-L6-v2`'s 256-token window rather than an arbitrary round number. A metadata header (fund name, CIK, accession number, filing date, period of report) is prepended to every chunk — this is what makes entity- and time-scoped filtering possible downstream without re-parsing.

---

## 3. Retrieval architecture

The retriever is hybrid by design because 13F queries split cleanly into two shapes: **exact-token lookups** (a CUSIP, a ticker) that BM25 nails and embeddings blur, and **semantic questions** ("which funds are concentrated in semiconductors") that embeddings handle and BM25 misses.

- **Hybrid search + RRF.** Stage 3 implements a `HybridRetriever` with Reciprocal Rank Fusion (`score += weight / (rrf_k + rank + 1)`, `rrf_k=60`, 50/50 weights); Stages 4–6 use LangChain's `EnsembleRetriever` with the same weighting.
- **Cross-encoder reranking.** `FlashrankRerank` sits *post-retrieval, pre-generation* via `ContextualCompressionRetriever`, with dual width — narrow (`top_n=8`) for entity-specific queries, wide (`top_n=20`) for cross-fund synthesis.
- **Post-retrieval precision filters.** Entity-scoped, temporal, and fund-diversity filters (detailed in the improvement log) run after reranking to fix the specific failures that reranking alone left on the table.

---

## 4. Generation & chain routing

Two chains, routed by query type:

- **Stuff chain** passes retrieved chunks *verbatim* into a single prompt. This is the right choice for fact/CUSIP/temporal lookups, where any summarization step would round or drop the exact value being asked for.
- **Map-Reduce** condenses each chunk before consolidating. It's reserved for genuinely cross-document questions (`Comparative / Deep Context`, `Sector Concentration`) where integrating across many funds matters more than verbatim fidelity.

A `MAP_REDUCE_TYPES` set dispatches at runtime; the chosen chain is recorded per query so the evaluation output is self-documenting. The tension between these two — verbatim fidelity vs. cross-document synthesis — recurs throughout the failure analysis below.

---

## 5. Evaluation methodology

### Baseline vs. enhanced, same judges

Every query runs twice: `run_baseline()` (the generator with **no** context) and `run_enhanced_rag()` (the full pipeline). Both answers are scored by the *same* dual-judge system, so the delta isolates what retrieval adds.

### Dual-judge, cross-vendor

Faithfulness is scored 1–5 on accuracy, specificity, and grounding by two judges from different organizations — GPT-4o-mini (OpenAI) and Gemini 2.5 Pro (Google) — and averaged (`score_dual()`). Using two vendors is deliberate: it removes the self-grading bias you'd get if the generator's own family also judged it. A keyword-count fallback scorer exists but only fires on API failure; across the reported run it never had to (0/20 evaluator fallbacks, 0/20 judge fallbacks).

### Grounding check

`grounding_check()` walks every `full-submission.txt` (up to 5MB each), extracts 9-character alphanumeric tokens containing a digit (the CUSIP pattern) from the response, and verifies each against the corpus. Crucially, it returns `None` (indeterminate) — not `True` — for refusals with no CUSIPs, so a refusal can't masquerade as "grounded." That distinction was itself one of the improvements (see #9).

### Headline result

Baseline **1.75/5** → enhanced **4.45/5** (**+2.70**). Grounding passed on 7/10 queries, with 2 indeterminate (justified refusals) and 1 known false-positive (Q4, dissected below). Per-query numbers are in the [README](README.md#results).

---

## 6. The retrieval-precision gap (the honest methodological hole)

Retrieval quality here is evaluated **indirectly** — via the grounding check (a CUSIP-verification proxy for precision) and the judges' faithfulness scores, plus a BM25 coverage diagnostic that confirms key entities are indexed. What's **missing** is formal information-retrieval metrics: there is no `precision@k`, `recall@k`, or MRR computed against a hand-labeled relevance set.

This is a real limitation, not a footnote. The grounding check tells me whether *cited* identifiers are real, but not whether the *right* chunks were retrieved and ranked ahead of distractors. A proper next step is to label a relevance set (query → known-relevant chunks) and compute standard IR metrics; that would turn statements like "reranking helped" into measured numbers rather than observed behavior. I flag it here because the rest of the evaluation is rigorous enough that this gap stands out.

---

## 7. The improvement log

The first end-to-end run was bad: avg faithfulness **1.4/5** baseline → **2.2/5** enhanced. Retrieval was working but generation was throwing the data away. Sixteen changes, grouped by the problem they solved, got it to 4.45. I'm keeping the failure observations attached to each fix because the observations are the transferable part.

### 7a. Getting the generator to use the context at all (1–6)

**1 — Generator swap.** 7/10 queries scored 1/5 *with* RAG active; judge rationales showed the generator fabricating data and ignoring retrieved chunks entirely — a classic small-quantized-model instruction-following failure. Swapping to a model that reliably copies verbatim from context and refuses when data is absent was the single biggest lever.

**2 — Chain routing (Stuff vs Map-Reduce).** Map-Reduce was being applied to *everything*. For single-fact lookups (Q01/Q02/Q05/Q08/Q10) its abstractive summarization is structurally wrong — it discards the exact share counts and CUSIPs the question asks for. Introduced `MAP_REDUCE_TYPES` so only genuinely cross-fund queries take that path; everything else goes Stuff.

**3 — Verbatim preservation.** Even with context present, prompts asking for "concise summaries" invited paraphrasing — rounded numbers, dropped CUSIPs, shortened fund names. Added a `CRITICAL` block to every template (Map, Reduce, Stuff system, Stuff user): *"Copy ALL numerical values, share counts, CUSIPs, dates, and fund names VERBATIM. Do NOT paraphrase, round, or restate."*

**4 — Query-type routing layer.** The `MAP_REDUCE_TYPES` set plus the `run_enhanced_rag()` dispatcher, with a `chain_label` recorded per query and surfaced in the results table (same mechanism as #2, formalized).

**5 — Coverage diagnostic.** Q01/Q02 (Berkshire's Apple position) scored 1/5 because the *retrieval* was failing — the correct chunks weren't coming back at all, so no generator change could help. Added a diagnostic cell that checks, before the eval loop, whether the BM25 corpus and Chroma store actually surface Berkshire+Apple chunks. If both warn, the root cause is upstream (Stage 2 XML namespace handling for CIK `0001067983`, or Stage 3 indexing) and those stages must be re-run. Making the failure visible *before* a full run saved a lot of wasted evaluation cycles.

**6 — Judge headroom.** The original GPT-4o judge (GitHub Models free tier) capped at 50 calls/day and 10/min; a single eval run needs 20 judge calls, so >2 runs/day hit the wall and dropped to the unreliable keyword fallback. Moving the judge to a higher-quota tier restored full LLM judging and let the inter-call delay drop from 7s to 4s.

> **Which models actually ship.** The final pipeline (Stages 4–6) runs **`gemini-3.1-flash-lite-preview`** as the generator (Google AI Studio), **`gpt-4o-mini`** as the evaluator (GitHub Models / OpenAI), and **`gemini-2.5-pro`** as the second judge (Google AI Studio). The improvement log above is *history*: earlier iterations passed through a local Ollama model (`qwen2.5:7b`) and GPT-family generators before settling on Gemini, so notes referencing "a 7B model" or `_call_ollama` describe intermediate stages that were later replaced. Where an improvement's reasoning still holds regardless of generator (prompt framing, chunk pre-filtering, entity/temporal filtering), it carried forward to the shipped configuration.

### 7b. Killing the refusal bias (7–11)

After 1–6, the enhanced pipeline was *still* scoring 1/5 on many queries even when `grounded=True`. Three interacting bugs, all around a small model over-refusing:

**7 — Positive prompt framing.** Stacked "Do NOT guess / Do NOT fabricate / Do NOT use outside knowledge" instructions read, to a 7B model, as "the safe move is to refuse" — even with the answer sitting in context. Reframed from prohibition to instruction: *"The filing data below contains the answer. Extract it directly. If and ONLY if the data point is truly absent from every row below, state..."* — which still permits correct refusal for genuinely missing data (Q06/Q07/Q09).

**8 — Extraction cue.** Added *"The data above contains the answer."* immediately before the answer instruction in every user prompt (Stuff/Map/Reduce) — a one-line prime toward extraction over refusal.

**9 — Non-vacuous grounding check.** The old check returned `(True, "no CUSIPs to verify")` for any response containing zero CUSIPs — which is *always* true of a refusal, so refusals looked "grounded." Added refusal-phrase detection: refusal + no CUSIPs now returns `None` (indeterminate), mapped to `N/A` in the results, and the failure-mode detector flags "refusal despite available context." This is what turned the refusal bug from invisible to measurable.

**10 — Token budget.** Raised the generator cap 600 → 1024 (Map steps stay at 400). At 600 tokens, tabular answers with several CUSIPs/share counts were truncating or compressing away specific values.

**11 — Large-chunk pre-filtering.** Berkshire's filing parsed as a *single* ~100-row chunk. Dropped whole into a 7B context window, the model couldn't locate the Apple rows among 100 others and gave up. `_prefilter_chunks()` splits a chunk into header + data rows, keeps only rows matching question keywords (>3 chars), and rebuilds header + matched rows — so Q01 sees ~12 Apple rows instead of 100+. Falls back to the full chunk if nothing matches.

### 7c. Retrieval precision fixes (12–16)

**12 — Entity-scoped filter.** Three regressions (enhanced *below* baseline) shared a cause: the retriever returned plausible chunks from the *wrong fund*, and a 7B model couldn't hold entity identity under that noise, so it cross-attributed holdings. The baseline, with no context, safely refused and scored better. This is a retrieval-precision problem wearing a generation-quality costume. Fix: an alias map built from corpus metadata (full names, parenthetical names like "Warren Buffett" → the Berkshire entity, two-word short forms), longest-match entity extraction from the question (empty set for broad "all funds / across all" queries), and a post-rerank filter keeping only target-fund chunks — with a fallback to unfiltered if that would empty the set.

**13 — Temporal filter.** Prevents temporal bleeding: a "Feb 2026" query shouldn't get 2025 chunks just because they're semantically similar. Filters on `period_of_report` first, `filing_date` only as fallback when the period is UNKNOWN.

**14 — Fund-diversity filter.** For cross-fund synthesis, `max_per_fund=2` stops a single fund's chunks from crowding out the others, so Map-Reduce actually sees multiple funds.

**15 — Dual-width reranker.** Narrow (`top_n=8`) for entity queries, wide (`top_n=20`) for synthesis — selected by query scope at runtime. (The retrieval-side complement to #12/#14.)

**16 — Map summary logging.** `run_map_reduce_rag()` logs every intermediate Map summary (excerpt number, fund, chunk index, preview) into the `map_summaries` field. This is partial mitigation for the risk in §8 — it makes intermediate steps *visible*, though not yet *audited*.

---

## 8. The unaudited map-step: a "hallucination laundering" risk

This is the limitation I find most interesting, so it gets its own section.

The audit trail captures the **final** reduce-step output — the consolidated report — with full provenance: retrieved chunks, accession numbers, relevance scores, and a sample of the Map prompt. End-to-end, a reader can trace the answer to specific SEC accession numbers.

But there's a structural gap. The **intermediate Map summaries** (one per retrieved chunk) are generated independently and fed into Reduce, and they are **not individually faithfulness-checked**. That opens a laundering path:

1. A Map summary invents or distorts a claim about a holding, fund, or share count.
2. Reduce treats that summary as a trustworthy input and folds it into the final report.
3. Because only the final output is audited, the hallucination's origin in the Map step is never flagged.

The verbatim instruction (#3) reduces the *likelihood* — a Map step told to copy values verbatim is less likely to invent them — but it doesn't *close* the hole, and #16's logging makes the intermediate steps visible without validating them.

**The production fix** is per-summary faithfulness verification: run the LLM-as-judge scorer on every Map summary against its source chunk before Reduce, discard anything below a threshold (say <4/5), and log every intermediate score. I didn't implement it here because it costs N extra judge calls per query (N = retrieved chunks), which is painful on CPU-only inference. That's an acceptable trade for a prototype and unacceptable for anything touching real regulatory filings — so it's the first thing I'd add in a production build.

---

## 9. Honest failure modes

**Q04 — grounding false-positive (score 3.0, grounding FAIL).** The model cited large numeric values (`433490391`, `300775438`) that are real `value` fields from the filing — but the grounding regex ("any 9-char alphanumeric with a digit = CUSIP candidate") swept them up as CUSIPs and failed to match them, flagging a hallucination that wasn't one. The lesson is that a heuristic grounding check collides with financial data whose dollar values are coincidentally 9 digits; a stricter CUSIP validator (checksum, position-aware extraction) would fix it.

**Q09 — flat cross-period result (score 3.0, delta 0.0).** The only query where enhanced didn't beat baseline. The temporal filter isolated the two periods correctly, but the model made listing errors and wrongly concluded there were no closed positions when the context was ambiguous. Cross-period queries route to Stuff, not Map-Reduce, and neither chain computes an explicit set-difference between periods. A dedicated "temporal diff" chain that literally diffs holdings across quarters would likely fix this class.

**Entity scoping is mitigated, not solved.** #12's alias matching handles the common cases but relies on aliases that can miss edge forms. The broad-retrieval-vs-precision-filtering tension (you need wide retrieval for synthesis and tight filtering for entity lookups) is exactly what the dual-width reranker (#15) juggles — well, but not perfectly.

**Sector questions have no native answer.** 13F filings carry no sector/industry field, so "which sector is fund X concentrated in" isn't answerable from the corpus alone. The fix is an external CUSIP/ticker → GICS (or SEC SIC) mapping table; without it, sector queries lean on the model inferring industry from issuer names, which is exactly the kind of ungrounded step this project exists to avoid.

---

## 10. Techniques inventory

| # | Technique | Where | What it does |
|---|-----------|-------|--------------|
| 1 | Hybrid BM25 + Vector retrieval | Stage 3–6 | RRF fusion of keyword and semantic retrievers |
| 2 | Cross-encoder reranking | Stage 4–6 | FlashrankRerank, dual-width (8 / 20) |
| 3 | Map-Reduce synthesis | Stage 4–6 | Parallel Map extraction + consolidating Reduce |
| 4 | Entity-scoped filtering | Stage 4–6 | Restricts chunks to target fund(s) |
| 5 | Temporal metadata filtering | Stage 4–6 | `period_of_report` filter; no cross-quarter bleed |
| 6 | Fund-diversity filter | Stage 4–6 | Caps chunks-per-fund for broad synthesis |
| 7 | Dual-judge evaluation | Stage 4–6 | Two vendors, no self-grading |
| 8 | Non-vacuous grounding check | Stage 4–6 | CUSIP verification vs. 5MB sources; refusals → indeterminate |
| 9 | Table-aware chunking | Stage 2 | Markdown tables, overlapping rows, per-chunk metadata |
| 10 | Chain routing | Stage 4–6 | Auto-dispatch Stuff vs Map-Reduce by query type |
| 11 | Chunk pre-filtering | Stage 4–6 | Keyword row filter for oversized tables |
| 12 | Routed reranker width | Stage 4–6 | Narrow for entity, wide for synthesis |

---

## 11. Reproducibility & secrets

The raw corpus (`Model/data/`) and the vector index (`Model/vector_db/`) are intentionally **not** committed — they're large and fully regenerable (Stage 1+2 rebuild the data, Stage 3 rebuilds the index; see the [README](README.md#regenerating-the-data)). Keeping them out of the repo also keeps the history small and avoids shipping a stale snapshot.

API credentials are handled by template: `gemini_api_key.example.txt` and `github_token.example.txt` ship with placeholder instructions, the real `*.txt` files are gitignored, and the notebooks read from the real files at runtime. That keeps live keys out of version control by construction rather than by remembering to scrub them.
