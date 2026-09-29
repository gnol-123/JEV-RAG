from __future__ import annotations
import os, sys, json, argparse
from dataclasses import dataclass
from typing import List
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.stats import pearsonr, spearmanr
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from typesafe_sdk import Noul, NoulCriteria, TypeSafeClient

from mas_survey.retreival_agent import RetrievalAgent, RetrievedDoc


JEV_MODEL   = "jev-latest"
MAX_WORKERS = 12


# DATACLASSES -----------------------------------------------------------------------

@dataclass
class ProbeEntry:
    question:  str
    doc_id:    str
    doc_rank:  int
    title:     str
    content:   str
    relevance: float   # ground truth 0 / 0.5 / 1 / 1.5 / 2


@dataclass
class EvalResult:
    variant:     str
    pearson_r:   float
    pearson_p:   float
    spearman_r:  float
    spearman_p:  float
    predicted:   List[float]
    true_scores: List[float]


# LOAD PROBE -----------------------------------------------------------------------

def load_probe(path: str) -> List[ProbeEntry]:

    with open(path) as f:
        raw = json.load(f)

    entries = []
    for item in raw:
        if item.get("relevance") is None:
            continue
        entries.append(ProbeEntry(
            question  = item["question"],
            doc_id    = item["doc_id"],
            doc_rank  = item["doc_rank"],
            title     = item.get("title", ""),
            content   = item.get("content_for_labeling", ""),
            relevance = float(item["relevance"]),
        ))

    return entries


# BASELINE VARIANTS -----------------------------------------------------------------------

def run_sparse_scores(agent: RetrievalAgent, entries: List[ProbeEntry]) -> List[float]:

    # ------------ Score each probe doc via BM25 ------------ #

    scores = []

    for entry in entries:
        results  = agent.sparse_search(entry.question, top_k=50)
        rank_map = {doc.doc_id: doc.score for doc in results}
        scores.append(rank_map.get(entry.doc_id, 0.0))

    return scores


def run_dense_scores(agent: RetrievalAgent, entries: List[ProbeEntry]) -> List[float]:

    # ------------ Score each probe doc via cosine similarity ------------ #

    scores = []

    for entry in entries:
        results  = agent.dense_search(entry.question, top_k=50)
        rank_map = {doc.doc_id: doc.score for doc in results}
        scores.append(rank_map.get(entry.doc_id, 0.0))

    return scores


def run_hybrid_scores(agent: RetrievalAgent, entries: List[ProbeEntry]) -> List[float]:

    # ------------ Score each probe doc via RRF fusion ------------ #

    scores = []

    for entry in entries:
        results  = agent.hybrid_search(entry.question, top_k=50)
        rank_map = {doc.doc_id: doc.score for doc in results}
        scores.append(rank_map.get(entry.doc_id, 0.0))

    return scores


# EMBEDDING RERANK VARIANT -----------------------------------------------------------------------

def run_embedding_rerank_scores(agent: RetrievalAgent, entries: List[ProbeEntry]) -> List[float]:

    # ------------ Direct cosine between query embed and doc embed ------------ #

    scores = []

    for entry in entries:

        if agent.embeddings is None or agent.doc_id_to_idx is None:
            scores.append(0.0)
            continue

        try:
            q_vec = np.array(agent.client.embed_one(entry.question), dtype=np.float32)

            if entry.doc_id in agent.doc_id_to_idx:
                idx     = agent.doc_id_to_idx[entry.doc_id]
                doc_vec = agent.embeddings[idx]
                score   = float(np.dot(doc_vec, q_vec))
            else:
                score = 0.0

            scores.append(score)

        except Exception as e:
            print(f"    Warning: embedding score failed for {entry.doc_id}: {e}")
            scores.append(0.0)

    return scores


# JEV VARIANT -----------------------------------------------------------------------

def _score_entry_jev(client: TypeSafeClient, entry: ProbeEntry) -> float:

    # ------------ BUILD STATE FOR JEV ------------ #

    state = {
        "query":    entry.question,
        "document": entry.content[:600] or entry.title or "(no content)",
    }

    relevance_q = Noul(
        instructions="Is the document relevant to the query?",
        criteria=NoulCriteria(
            true="The document directly addresses or provides evidence about the query topic.",
            false="The document is off-topic or only tangentially related.",
        ),
    )

    try:
        # ------------ CALL JEV ------------ #

        response = client.system_one(
            state=state,
            questions={"relevant": relevance_q},
            model=JEV_MODEL,
        )

        return response.answers["relevant"].noul

    except Exception as e:
        print(f"    Warning: Jev failed for {entry.doc_id}: {e}")
        return 0.0


def run_jev_scores(client: TypeSafeClient, entries: List[ProbeEntry]) -> List[float]:

    # ------------ Score all probe entries in parallel ------------ #

    print(f"    Scoring {len(entries)} probe entries with Jev ({MAX_WORKERS} workers)...")

    scores = [0.0] * len(entries)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(_score_entry_jev, client, entry): i
            for i, entry in enumerate(entries)
        }

        for future in as_completed(futures):
            i = futures[future]
            try:
                scores[i] = future.result()
            except Exception as e:
                print(f"    Warning: future failed: {e}")

    return scores


# METRICS -----------------------------------------------------------------------

def compute_metrics(variant: str, predicted: List[float], true_scores: List[float]) -> EvalResult:

    p_r, p_p = pearsonr(predicted, true_scores)
    s_r, s_p = spearmanr(predicted, true_scores)

    return EvalResult(
        variant    = variant,
        pearson_r  = round(float(p_r), 6),
        pearson_p  = round(float(p_p), 6),
        spearman_r = round(float(s_r), 6),
        spearman_p = round(float(s_p), 6),
        predicted  = predicted,
        true_scores= true_scores,
    )


def interpret_r(r: float) -> str:
    a = abs(r)
    if a >= 0.7: return "Strong"
    if a >= 0.4: return "Moderate"
    if a >= 0.2: return "Weak"
    return "Negligible"


# MAIN -----------------------------------------------------------------------

def _load_env(path: str = ".env"):
    env = Path(__file__).parent.parent / path
    if not env.exists():
        return
    with open(env) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",   default="config.yaml")
    p.add_argument("--probe",    default="experiments/probes/ir_probe.json")
    p.add_argument("--out",      default="eval/results/eval_results.json")
    p.add_argument("--variants", nargs="+",
                   default=["sparse", "dense", "hybrid", "embedding_rerank", "jev"],
                   help="Variants to run")
    return p.parse_args()


def main():
    _load_env()
    args = parse_args()

    # ------------ Load probe ------------ #
    print(f"Loading probe from {args.probe}...")
    entries     = load_probe(args.probe)
    true_scores = [e.relevance for e in entries]
    print(f"  {len(entries)} labeled entries")

    # ------------ Init agents ------------ #
    print("Initializing RetrievalAgent...")
    agent = RetrievalAgent(config_path=args.config)

    jev_client = None
    if "jev" in args.variants:
        typesafe_key = os.environ.get("TYPESAFE_API_KEY", "")
        if not typesafe_key:
            print("WARNING: TYPESAFE_API_KEY not set — skipping jev variant")
            args.variants = [v for v in args.variants if v != "jev"]
        else:
            jev_client = TypeSafeClient(api_key=typesafe_key)

    # ------------ Run ------------ #
    VARIANT_MAP = {
        "sparse":           lambda: run_sparse_scores(agent, entries),
        "dense":            lambda: run_dense_scores(agent, entries),
        "hybrid":           lambda: run_hybrid_scores(agent, entries),
        "embedding_rerank": lambda: run_embedding_rerank_scores(agent, entries),
        "jev":              lambda: run_jev_scores(jev_client, entries),
    }

    all_results: List[EvalResult] = []

    for variant in args.variants:
        if variant not in VARIANT_MAP:
            print(f"Unknown variant '{variant}', skipping")
            continue

        print(f"\n--- {variant.upper()} ---")

        predicted = VARIANT_MAP[variant]()
        result    = compute_metrics(variant, predicted, true_scores)
        all_results.append(result)

        print(f"  Pearson  r={result.pearson_r:.4f}  p={result.pearson_p:.4f}  [{interpret_r(result.pearson_r)}]")
        print(f"  Spearman r={result.spearman_r:.4f}  p={result.spearman_p:.4f}  [{interpret_r(result.spearman_r)}]")

    # ------------ Summary table ------------ #
    print("\n" + "=" * 60)
    print(f"{'Variant':<20} {'Pearson r':>10} {'Spearman r':>12}  Interpretation")
    print("-" * 60)
    for r in sorted(all_results, key=lambda x: x.pearson_r, reverse=True):
        print(f"{r.variant:<20} {r.pearson_r:>10.4f} {r.spearman_r:>12.4f}  {interpret_r(r.pearson_r)}")

    # ------------ Save ------------ #
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump({
            "probe":     args.probe,
            "n_entries": len(entries),
            "results": [
                {
                    "variant":       r.variant,
                    "pearson_r":     r.pearson_r,
                    "pearson_p":     r.pearson_p,
                    "spearman_r":    r.spearman_r,
                    "spearman_p":    r.spearman_p,
                    "interpretation": interpret_r(r.pearson_r),
                    "predicted":     r.predicted,
                    "true_scores":   r.true_scores,
                }
                for r in all_results
            ],
        }, f, indent=2)

    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
