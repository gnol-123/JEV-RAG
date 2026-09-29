from __future__ import annotations
import os, sys, json, yaml, argparse
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent))

from typesafe_sdk import Noul, NoulCriteria, TypeSafeClient

from mas_survey.retreival_agent import RetrievalAgent, RetrievedDoc
from utils.query_utils import load_query_json


JEV_MODEL   = "jev-latest"
CANDIDATE_K = 1000   # docs pulled from retrieval before Jev scoring
FINAL_K     = 100    # top docs returned after Jev rerank
MAX_WORKERS = 12     # parallel Jev calls


# DATACLASSES -----------------------------------------------------------------------

@dataclass
class JevScoredDoc:
    doc: RetrievedDoc
    confidence: float   # Jev noul score 0-1


@dataclass
class JevReranker:

    config_path: str = "config.yaml"
    retrieval_agent: Optional[RetrievalAgent] = field(default=None, repr=False)
    client: Optional[TypeSafeClient] = field(default=None, repr=False)

    def __init__(self, config_path: str = "config.yaml"):

        # --------------------- INIT RETRIEVAL ---------------------- #
        self.retrieval_agent = RetrievalAgent(config_path=config_path)

        # --------------------- INIT JEV CLIENT --------------------- #
        api_key = os.environ.get("TYPESAFE_API_KEY", "")
        if not api_key:
            raise RuntimeError("TYPESAFE_API_KEY not set.")

        self.client = TypeSafeClient(api_key=api_key)


    # SCORE SINGLE DOC ------------------------------------------------------------------

    def _score_doc(self, query: str, doc: RetrievedDoc) -> JevScoredDoc:

        # ------------ BUILD STATE FOR JEV ------------ #
        doc_text = " ".join(filter(None, [
            doc.title,
            doc.description,
            doc.post_content,
            doc.content,
        ]))[:600]

        state = {
            "query":    query,
            "document": doc_text or "(no content)",
        }

        # ------------ BUILD NOUL QUESTION ------------ #
        relevance_q = Noul(
            instructions="Is the document relevant to the query?",
            criteria=NoulCriteria(
                true="The document directly addresses or provides evidence about the query topic.",
                false="The document is off-topic or only tangentially related.",
            ),
        )

        try:
            # ------------ CALL JEV ------------ #
            response = self.client.system_one(
                state=state,
                questions={"relevant": relevance_q},
                model=JEV_MODEL,
            )

            confidence = response.answers["relevant"].noul

        except Exception as e:
            print(f"    Warning: Jev scoring failed for {doc.doc_id}: {e}")
            confidence = 0.0

        return JevScoredDoc(doc=doc, confidence=confidence)


    # RERANK -----------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        method: str = "hybrid",
        candidate_k: int = CANDIDATE_K,
        final_k: int = FINAL_K,
    ) -> List[JevScoredDoc]:

        # ------------ PHASE 1: CANDIDATE RETRIEVAL ------------ #
        print(f"  Phase 1: Retrieving top {candidate_k} candidates via {method}...")

        if method == "hybrid":
            candidates = self.retrieval_agent.hybrid_search(query, top_k=candidate_k)
        elif method == "dense":
            candidates = self.retrieval_agent.dense_search(query, top_k=candidate_k)
        elif method == "sparse":
            candidates = self.retrieval_agent.sparse_search(query, top_k=candidate_k)
        else:
            candidates = self.retrieval_agent.hybrid_search(query, top_k=candidate_k)

        print(f"    Retrieved {len(candidates)} candidates")

        if not candidates:
            return []

        # ------------ PHASE 2: JEV SCORING (PARALLEL) ------------ #
        print(f"  Phase 2: Scoring {len(candidates)} docs with Jev ({MAX_WORKERS} workers)...")

        scored: List[JevScoredDoc] = []

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {
                pool.submit(self._score_doc, query, doc): doc
                for doc in candidates
            }

            for i, future in enumerate(as_completed(futures), 1):
                try:
                    scored.append(future.result())
                except Exception as e:
                    doc = futures[future]
                    print(f"    Warning: future failed for {doc.doc_id}: {e}")
                    scored.append(JevScoredDoc(doc=doc, confidence=0.0))

                if i % 100 == 0:
                    print(f"    Scored {i}/{len(candidates)}")

        # ------------ PHASE 3: SORT & RETURN TOP-K ------------ #
        print(f"  Phase 3: Sorting by confidence...")

        scored.sort(key=lambda x: x.confidence, reverse=True)

        top_conf = scored[0].confidence if scored else 0
        bot_conf = scored[-1].confidence if scored else 0
        print(f"  Done. Top: {top_conf:.4f}  |  Bottom: {bot_conf:.4f}")

        return scored[:final_k]


# MAIN -----------------------------------------------------------------------

def _load_env(path: str = ".env"):
    if not Path(path).exists():
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--query",  default=None, help="Single query string (skips config questions)")
    p.add_argument("--method", default="hybrid", choices=["hybrid", "dense", "sparse"])
    p.add_argument("--out",    default="experiments/jev_reranking.json")
    return p.parse_args()


def main():
    _load_env()
    args = parse_args()

    if not os.environ.get("TYPESAFE_API_KEY"):
        print("ERROR: TYPESAFE_API_KEY not set. Add it to .env")
        sys.exit(1)

    # ------------ LOAD QUERIES ------------ #
    if args.query:
        queries = [{"qid": 0, "query": args.query, "distribution": {}}]
    else:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)

        q_path = cfg["data"]["questions_path"]
        raw_queries = load_query_json(q_path)
        queries = [{"qid": q.qid, "query": q.query, "distribution": q.distribution} for q in raw_queries]

    print(f"Loaded {len(queries)} queries")

    # ------------ INIT RERANKER ------------ #
    print("Initializing JevReranker...")
    reranker = JevReranker(config_path=args.config)
    print("Ready.\n")

    # ------------ RUN ------------ #
    all_results = []

    for entry in queries:
        qid   = entry["qid"]
        query = entry["query"]

        print(f"\n[{qid}] {query[:80]}")

        ranked = reranker.rerank(query, method=args.method)

        result = {
            "qid":    qid,
            "query":  query,
            "method": args.method,
            "ranked": [
                {
                    "rank":       rank,
                    "doc_id":     r.doc.doc_id,
                    "title":      r.doc.title,
                    "confidence": r.confidence,
                    "source":     r.doc.source,
                }
                for rank, r in enumerate(ranked, 1)
            ],
        }

        all_results.append(result)

        # ------------ PRINT TOP 5 ------------ #
        print(f"  Top 5:")
        for r in ranked[:5]:
            print(f"    [{r.doc.doc_id}] conf={r.confidence:.4f}  {r.doc.title[:60]}")

    # ------------ SAVE ------------ #
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump({"results": all_results}, f, indent=2)

    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
