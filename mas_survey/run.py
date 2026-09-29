from __future__ import annotations
import os, sys, json, csv, yaml, argparse
from pathlib import Path

# ------------ LOAD ENV BEFORE OTHER IMPORTS ------------ #
_env = Path(__file__).parent.parent / ".env"
if _env.exists():
    with open(_env) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

sys.path.insert(0, str(Path(__file__).parent.parent))

from mas_survey.ingest import load_qeury_from_arg
from mas_survey.retreival_agent import RetrievalAgent, RetrievedDoc
from mas_survey.prediction_agent import PredictionAgent
from mas_survey.query_expansion_agent import QueryExpansionAgent
from mas_survey.jev_reranking import JevReranker


# VARIANTS -----------------------------------------------------------------------
# Each variant defines retrieval method + reranking stage.
# candidate_k = pool size fed into reranker.
# rerank_top_k = how many docs the reranker returns for prediction.

VARIANTS = {
    "sparse":            {"method": "sparse",  "rerank": None,        "candidate_k": 100,  "rerank_top_k": 15},
    "dense":             {"method": "dense",   "rerank": None,        "candidate_k": 100,  "rerank_top_k": 15},
    "hybrid":            {"method": "hybrid",  "rerank": None,        "candidate_k": 100,  "rerank_top_k": 15},
    "hybrid_emb_rerank": {"method": "hybrid",  "rerank": "embedding", "candidate_k": 300,  "rerank_top_k": 15},
    "hybrid_llm_rerank": {"method": "hybrid",  "rerank": "llm",       "candidate_k": 100,  "rerank_top_k": 10},
    "hybrid_jev_rerank": {"method": "hybrid",  "rerank": "jev",       "candidate_k": 500,  "rerank_top_k": 15},
}


# LOAD ENV -----------------------------------------------------------------------

# RETRIEVAL -----------------------------------------------------------------------

def get_supports(
    query: str,
    distribution: dict,
    agent: RetrievalAgent,
    jev_reranker: JevReranker,
    variant_cfg: dict,
) -> tuple[list[str], list[RetrievedDoc]]:

    import time
    method       = variant_cfg["method"]
    rerank       = variant_cfg["rerank"]
    candidate_k  = variant_cfg["candidate_k"]
    rerank_top_k = variant_cfg["rerank_top_k"]

    # ------------ RETRIEVE CANDIDATES ------------ #

    t0 = time.time()
    print(f"    [retrieve] method={method}  candidate_k={candidate_k} ...")

    if method == "sparse":
        docs = agent.sparse_search(query, top_k=candidate_k)
    elif method == "dense":
        docs = agent.dense_search(query, top_k=candidate_k)
    else:
        docs = agent.hybrid_search(query, top_k=candidate_k)

    print(f"    [retrieve] got {len(docs)} docs in {time.time()-t0:.1f}s")

    if not docs:
        print(f"    [retrieve] EMPTY — skipping rerank")
        return [], []

    # ------------ RERANK ------------ #

    t0 = time.time()
    if rerank == "embedding":
        print(f"    [rerank:embedding] scoring {len(docs)} docs → top {rerank_top_k}...")
        top_docs = agent.embedding_rerank(query, distribution, docs, top_k=rerank_top_k)

    elif rerank == "llm":
        print(f"    [rerank:llm] scoring {len(docs)} docs → top {rerank_top_k}...")
        top_docs = agent.llm_rerank(query, distribution, docs, top_k=rerank_top_k)

    elif rerank == "jev":
        print(f"    [rerank:jev] scoring {candidate_k} docs → top {rerank_top_k}...")
        jev_scored = jev_reranker.rerank(query, method=method, candidate_k=candidate_k, final_k=rerank_top_k)
        top_docs   = [s.doc for s in jev_scored]

    else:
        print(f"    [rerank:none] taking top {rerank_top_k} as-is")
        top_docs = docs[:rerank_top_k]

    if rerank:
        print(f"    [rerank] done in {time.time()-t0:.1f}s → {len(top_docs)} top docs")

    # ------------ BUILD 100 SUPPORTS ------------ #

    support_seen = set()
    support_docs = []

    # --- top reranked docs first --- #
    for doc in top_docs:
        if doc.doc_id not in support_seen:
            support_docs.append(doc)
            support_seen.add(doc.doc_id)

    # --- fill remainder from candidate pool --- #
    for doc in docs:
        if len(support_docs) >= 100:
            break
        if doc.doc_id not in support_seen:
            support_docs.append(doc)
            support_seen.add(doc.doc_id)

    # --- random pad if still short --- #
    import numpy as np
    if len(support_docs) < 100 and agent.doc_ids is not None:
        rng       = np.random.RandomState(42)
        available = [d for d in agent.doc_ids if d not in support_seen]
        if available:
            padding = rng.choice(available, size=min(100 - len(support_docs), len(available)), replace=False)
            for pad_id in padding:
                support_docs.append(RetrievedDoc(doc_id=pad_id, title="", content="", score=0.0, source="padding"))

    support_ids = [doc.doc_id for doc in support_docs[:100]]
    return support_ids, top_docs


# SAVE -----------------------------------------------------------------------

def save_results(variant: str, all_results: list, out_dir: Path):

    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------ JSON ------------ #
    json_path = out_dir / f"{variant}.json"
    with open(json_path, "w") as f:
        json.dump({"variant": variant, "results": all_results}, f, indent=2)

    # ------------ CSV ------------ #
    csv_path = out_dir / f"{variant}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["question", "distribution", "supports"])

        for r in all_results:
            distribution = r.get("predicted_distribution", {})
            supports     = r["supports"]

            # ------------ PAD / TRIM TO EXACTLY 100 ------------ #
            if len(supports) < 100:
                supports = supports + [f"pad_{i}" for i in range(len(supports), 100)]
            else:
                supports = supports[:100]

            writer.writerow([r["query"], json.dumps(distribution), json.dumps(supports)])

    print(f"    Saved → {json_path.name}  {csv_path.name}")


# MAIN -----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Run all retrieval experiment variants")
    p.add_argument("--config",   default="config.yaml")
    p.add_argument("--variants", nargs="+", default=list(VARIANTS.keys()),
                   help="Variants to run (default: all)")
    p.add_argument("--out_dir",  default="experiments/runs",
                   help="Output directory for results")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.environ.get("TOGETHER_API_KEY"):
        print("ERROR: TOGETHER_API_KEY not set. Add it to .env")
        sys.exit(1)

    # ------------ LOAD CONFIG + QUERIES ------------ #
    queries = load_qeury_from_arg(args)
    print(f"Loaded {len(queries)} queries")

    # ------------ INIT AGENTS ------------ #
    print("Initializing agents...")

    expansion_agent = QueryExpansionAgent()

    retrieval_agent = RetrievalAgent(config_path=args.config)
    retrieval_agent.expansion_agent = expansion_agent

    pred_agent = PredictionAgent()

    # ------------ INIT JEV (optional) ------------ #
    jev_reranker = None
    if any(VARIANTS[v]["rerank"] == "jev" for v in args.variants if v in VARIANTS):
        typesafe_key = os.environ.get("TYPESAFE_API_KEY", "")
        if not typesafe_key:
            print("WARNING: TYPESAFE_API_KEY not set — skipping jev variants")
            args.variants = [v for v in args.variants if VARIANTS.get(v, {}).get("rerank") != "jev"]
        else:
            jev_reranker = JevReranker(config_path=args.config)

    out_dir = Path(args.out_dir)

    # ------------ RUN VARIANTS ------------ #

    for variant in args.variants:

        if variant not in VARIANTS:
            print(f"Unknown variant '{variant}', skipping")
            continue

        cfg = VARIANTS[variant]
        print(f"\n{'='*60}")
        print(f"VARIANT: {variant.upper()}")
        print(f"  method={cfg['method']}  rerank={cfg['rerank']}  candidate_k={cfg['candidate_k']}")
        print(f"{'='*60}")

        all_results = []

        for i, qery in enumerate(queries, 1):

            print(f"\n  [{i}/{len(queries)}] {qery.query[:70]}")

            question     = qery.query
            distribution = qery.distribution

            # ------------ RETRIEVE ------------ #
            supports, top_docs = get_supports(
                question, distribution,
                retrieval_agent, jev_reranker,
                cfg
            )

            print(f"    Supports: {len(supports)}  Top docs: {len(top_docs)}")

            # ------------ PREDICT DISTRIBUTION ------------ #
            answer_options = list(distribution.keys()) if distribution else []
            pred_distribution = {}

            if answer_options:
                import time
                t0 = time.time()
                print(f"    [predict] {len(answer_options)} options, {len(top_docs)} top docs → LLM call...")
                pred_distribution = pred_agent.predict_distribution(qery, answer_options, top_docs)
                print(f"    [predict] done in {time.time()-t0:.1f}s")
                print(f"    Distribution: { {k: round(v,3) for k,v in pred_distribution.items()} }")

            # ------------ FORMAT ------------ #
            result = {
                "qid":                  qery.qid,
                "query":                question,
                "variant":              variant,
                "supports":             supports,
                "predicted_distribution": pred_distribution,
                "num_supports":         len(supports),
                "top_docs_preview": [
                    {"rank": r, "doc_id": d.doc_id, "score": round(float(d.score), 4), "title": d.title}
                    for r, d in enumerate(top_docs[:5], 1)
                ],
            }

            all_results.append(result)

        # ------------ SAVE ------------ #
        print(f"\n  Saving {variant}...")
        save_results(variant, all_results, out_dir)

    # ------------ SUMMARY ------------ #
    print(f"\n{'='*60}")
    print(f"DONE — {len(args.variants)} variant(s) × {len(queries)} queries")
    print(f"Output: {out_dir}/")
    for v in args.variants:
        if v in VARIANTS:
            print(f"  {v}.json / {v}.csv")


if __name__ == "__main__":
    main()
