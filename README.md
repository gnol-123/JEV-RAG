# RAG Using TypeSafe AI's JEV

Retrieval + prediction pipeline for a public-opinion survey task. Six retrieval variants run head-to-head, including one that reranks with TypeSafe AI's Jev (a System One classifier that returns typed decisions instead of text).

Corpus: 230,196 docs. Dev set: 15 queries. Mini dev: 1 query for smoke tests.

## Setup

1. Python 3.13, Windows or Linux. A CUDA GPU is optional but strongly recommended for embedding.
2. Install deps:
   ```
   pip install -r requirements.txt
   ```
3. Download the large artifacts (documents, sparse index, embeddings). Too big for GitHub. Grab from Google Drive: **<PASTE DRIVE LINK HERE>**. Unzip so you get:
   ```
   data/documents.jsonl
   artifacts/id_to_embedding.npz
   artifacts/index_Mas_JS/           (Whoosh sparse index folder)
   ```
   If you skip this, run `python -m index.build --embed-cuda` to rebuild both from the mini set or your own corpus.
4. If you want GPU embedding, install a CUDA torch wheel matching your driver. For CUDA 12.4 on Windows py3.13:
   ```
   pip install torch --index-url https://download.pytorch.org/whl/cu124
   ```
5. Copy `.env.example` to `.env` and fill in keys:
   - `TOGETHER_API_KEY` — chat model (required)
   - `TYPESAFE_API_KEY` — Jev reranker (required for `hybrid_jev_rerank` variant)
   - `OPENAI_API_KEY` — only needed if you embed via `--embed-api` instead of local GPU

## Quick start

Build the sparse index and embed the corpus:

```
python -m index.build --embed-cuda      # local GPU embed (recommended)
# or:
python -m index.build --embed-api       # OpenAI embed
# or:
python -m index.build                   # sparse only, no embed
```

Set `$env:EMBED_YES="1"` to skip the interactive y/n prompt.

Run the pipeline (all 6 variants):

```
python -m mas_survey.run --out_dir experiments/runs
```

Single variant only:

```
python -m mas_survey.run --variants hybrid_jev_rerank
```

Results land in `experiments/runs/<variant>.json` and `<variant>.csv`.

## Variants

Each variant retrieves 100 support documents per query and picks the top 15 for the prediction agent.

| Variant | Retrieval | Rerank | Notes |
|---|---|---|---|
| `sparse` | BM25 (Whoosh) | none | Query is cleaned: stopwords stripped, first 5 content words kept. |
| `dense` | bge-base cosine | none | 768-dim embeddings, normalized. |
| `hybrid` | sparse + dense with RRF fusion (k=60) | none | Default alpha 0.6 favors sparse. |
| `hybrid_emb_rerank` | hybrid top 300 | Cosine of query embed vs doc embed | Cheap, adds distribution keywords to query embed. |
| `hybrid_llm_rerank` | hybrid top 100 | Together LLM scores doc batches of 15 | Slower, higher quality. |
| `hybrid_jev_rerank` | hybrid top 500 | Jev Noul per (query, doc) with 12 parallel workers | Typed confidence 0-1, sorted, top 15 returned. |

Prediction: `Llama-3.3-70B-Instruct-Turbo` on Together, JSON-mode, temperature 0.3.

## Costs and time

### One-time index build (230k docs)

| Step | Time | $ |
|---|---|---|
| Sparse Whoosh index | ~5 min, CPU | 0 |
| Embed, local GPU (RTX 4070, bge-base fp16, batch 128) | ~8-12 min | 0 |
| Embed, OpenAI `text-embedding-3-small` at tier 1 | ~50 min (rate-limited) | ~$1 |

Estimates. Local GPU is bottlenecked by other apps holding VRAM (kill LM Studio, Copilot, and heavy browser tabs first).

### Per query, per variant (rough)

Query embeds are ~50 tokens. Doc text passed to LLMs and Jev is truncated to 600 chars.

| Variant | Embed calls | Rerank cost | Predict cost |
|---|---|---|---|
| `sparse` | 0 | 0 | ~$0.0005 |
| `dense` | 1 | 0 | ~$0.0005 |
| `hybrid` | 1 | 0 | ~$0.0005 |
| `hybrid_emb_rerank` | 2 | 0 | ~$0.0005 |
| `hybrid_llm_rerank` | 1 | ~$0.001 (Together) | ~$0.0005 |
| `hybrid_jev_rerank` | 1 | ~$0.005 (500 Jev calls × ~250 tok × $0.042/M) | ~$0.0005 |

Jev pricing from TypeSafe: $0.042 per million input tokens, output free.

### Full dev run (15 queries × 6 variants)

| Source | Total |
|---|---|
| Jev rerank | ~$0.08 |
| Together LLM rerank + prediction | ~$0.05 |
| OpenAI query embeds (if using API embed) | negligible |
| **Total API** | **~$0.13** |

Wall clock, rough: 10-15 min end to end. Jev calls are parallelized 12 wide; the LLM rerank is the main sequential bottleneck.

Mini smoke test (1 query × 6 variants): ~1 min, under $0.01.

## Results (15 dev queries)

Measured on the RTX 4070 setup above, one pass, no retries. Recall is against the labeled `dev_groundtruth.json` supports (100 human-picked per query). TVD is total variation distance between predicted answer distribution and ground truth (lower is better, 0 = perfect, 1 = disjoint).

| Variant | Wall clock (total) | Retrieve avg | Rerank avg | Predict avg | Recall@100 | TVD |
|---|---:|---:|---:|---:|---:|---:|
| `sparse` | 53s | 2.2s | — | 1.3s | 0.079 | 0.258 |
| `dense` | 52s | 2.0s | — | 1.5s | 0.086 | 0.335 |
| `hybrid` | 61s | 2.7s | — | 1.3s | 0.084 | 0.304 |
| `hybrid_emb_rerank` | 74s | 3.3s | 0.1s | 1.5s | 0.084 | 0.312 |
| `hybrid_llm_rerank` | 346s | 4.1s | 17.7s | 1.2s | 0.084 | **0.252** |
| `hybrid_jev_rerank` | 316s | 4.5s | 15.1s | 1.5s | 0.086 | 0.334 |

Takeaways from this run:

- Recall@100 clusters around 8-9% across all variants. The dev groundtruth marks 100 specific supports per query; the pipeline finds different but topically relevant docs, so overlap is modest. Recall would climb with more human labels or a wider top-K.
- On the prediction task itself (TVD), `hybrid_llm_rerank` edges out the others by a small margin, with plain `sparse` a close second. Jev-reranked results were no better than raw hybrid on this 15-query set. Jev has a real speed win over LLM rerank (15.1s vs 17.7s average, 12 parallel workers vs sequential batches).
- The three cheap variants (sparse/dense/hybrid) finish the whole dev set in under a minute each. The two LLM-based rerankers dominate wall time.
- One query set is small. A larger set is needed to say anything definitive about Jev vs LLM rerank quality.

## Layout

```
mas_survey/
  retreival_agent.py       sparse, dense, hybrid + embedding/LLM rerank
  jev_reranking.py         Jev variant, called by run.py
  query_expansion_agent.py LLM-driven query variations
  prediction_agent.py      final distribution predictor
  ingest.py                query loader
  run.py                   entry point for all 6 variants
index/
  build.py                 CLI: --embed-cuda / --embed-api / (bare) sparse only
  whoosh_index.py          BM25 index builder
  embed.py                 batched embedder with checkpoint/resume
  ingest.py                jsonl loader with optional subchunking
eval/
  eval.py                  probe-based correlation eval (Pearson + Spearman)
utils/
  llm_client.py            Together (chat) + local sentence-transformers or OpenAI (embed)
  utils.py, query_utils.py loaders
data/
  documents.jsonl          full corpus (230k)
  mini_documents.jsonl     smoke test corpus
  dev/dev.json             15 dev queries
  dev/mini_dev.json        1-query smoke set
  dev/test.json            held-out
artifacts/
  index_Mas_JS/            Whoosh sparse index
  id_to_embedding.npz      dense embeddings
  embed_checkpoint.jsonl   incremental checkpoint (safe to delete after .npz is built)
experiments/
  runs/                    per-variant JSON + CSV output
  probes/ir_probe.json     labeled relevance probe for eval.py
```

## Config

`config.yaml` controls corpus/query paths and index locations.

```yaml
seed: 42

data:
  documents_path: ./data/documents.jsonl
  questions_path: ./data/dev/dev.json
  test_path:      ./data/dev/test.json

index:
  output_dir: ./artifacts/index_Mas_JS
  emb_dir:    ./artifacts

run:
  output_csv:  ./results/submission.csv
  supports_k:  100
```

Swap `documents.jsonl` for `mini_documents.jsonl` when iterating.

## Troubleshooting

**"Unknown model jev-1.12"**: TypeSafe rolled Jev to 1.13. `jev-latest` alias tracks the current version and is what the code uses.

**"Server overloaded" from Together on embed**: `togethercomputer/m2-bert-80M-32k-retrieval` is decommissioned. Together's serverless embedding endpoints are gone. Use `--embed-cuda` (local) or `--embed-api` (OpenAI).

**Dense search returns 0 docs**: Likely `TOGETHER_API_KEY` (or `OPENAI_API_KEY` if using API embed) not visible at import time. The pipeline loads `.env` before importing agents, but if you call `RetrievalAgent` directly in a script, load `.env` first.

**Embed at 8 docs/s on GPU**: Another process is holding VRAM. `nvidia-smi` on Windows underreports per-process memory, so kill obvious suspects: LM Studio, Ollama, Copilot, Chrome/Edge tabs with heavy GPU work.

**"Invalid model ID" from OpenAI**: `.env` has `FACTCHECK_EMBED_MODEL` set to something the OpenAI endpoint doesn't recognize (an HF path or the old Together model). Delete the line or set it to `text-embedding-3-small`.

**Embed crashed mid-run**: Just re-run the same command. `artifacts/embed_checkpoint.jsonl` is written per batch, and `embed_docs` skips any doc id already in the checkpoint.

**Sparse index has 0 docs**: Older `MAIN_WRITELOCK` file left behind by a killed builder. `Remove-Item artifacts/index_Mas_JS -Recurse -Force` and rebuild.
