from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from pathlib import Path
import yaml, sys
import numpy as np
from whoosh import index
from whoosh.qparser import MultifieldParser, OrGroup
from whoosh.query import Term

from nltk.corpus import stopwords as _stopwords
_STOP = set(_stopwords.words("english"))


def _clean_query(query: str, top_n: int = 5) -> str:
    # ------ drop stopwords + punctuation, keep first top_n content words ------ #
    toks = [w.strip(".,?!:;\"'()[]") for w in query.split()]
    toks = [w for w in toks if w and w.lower() not in _STOP and any(c.isalnum() for c in w)]
    return " ".join(toks[:top_n]) if toks else query

from utils.llm_client import LLMClient

from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


DATA_CONFIG_DIR = "data"
INDEX_CONFIG_DIR = "index"


@dataclass
class RetrievedDoc:
    doc_id: str
    title: str
    content: str
    score: float
    source: str = ""
    description: str = ""
    post_content: str = ""


@dataclass
class RetrievalAgent:

    config_path: str = "config.yaml"
    client: Optional[LLMClient] = None
    sparse_index: Any = None
    embeddings: Optional[np.ndarray] = None
    doc_ids: Optional[np.ndarray] = None
    doc_id_to_idx: Optional[Dict[str, int]] = None

    def __init__(self, config_path: str = "config.yaml"):

        try:
            self.client = LLMClient()
        except:
            self.client = None

        # ------- OPEN CONFIG -------------- #
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # ---------------- Load whoosh index ---------------------- #
        index_dir = self.config[INDEX_CONFIG_DIR]["output_dir"]

        if Path(index_dir).exists():
            try:
                self.sparse_index = index.open_dir(index_dir)
            except:
                print(f"Warning: Failed to get sparse index from {index_dir}. Please rebuild")

        # ---------------- Load embeddings ---------------------- #
        emb_dir = self.config.get(INDEX_CONFIG_DIR, {}).get("emb_dir")

        if not emb_dir:
            print(f"Error: directory not specified in config under '{INDEX_CONFIG_DIR}'")
            sys.exit(-1)

        emb_dir_path = Path(emb_dir)
        emb_dir_path.mkdir(parents=True, exist_ok=True)

        emb_path = emb_dir_path / "id_to_embedding.npz"
        if emb_path.exists():
            try:
                data = np.load(emb_path, allow_pickle=True)
                self.embeddings = data["embeddings"]
                self.doc_ids = data["ids"]

                # ------------------------- MAP ID to INDEX ---------------------------- #
                self.doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(self.doc_ids)}
            except:
                raise FileExistsError(f"Failed to get embedding from {emb_path}")
        else:
            print("No embeddings. Please rebuild...")
            sys.exit(-1)


    # RERANKERS -------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def embedding_rerank(self, query: str, distribution: dict, docs: List[RetrievedDoc], top_k: int = 100) -> List[RetrievedDoc]:

        # -------------- CHECK IF EMBEDDING IS AVAILABLE --------------- #
        if self.embeddings is None:
            print("Warning: no embeddings defaulting...")
            return docs[:top_k]

        try:
            # ------ add distribution to query --------- #
            query_and_dist = query

            if distribution:
                answers = " ".join(distribution.keys())
                query_and_dist = f"{query} {answers}"

            # ------------------- Vectorize Query ------------ #
            query_vec = np.array(self.client.embed_one(query_and_dist), dtype=np.float32)

            reranked = []

            # ------------------ GET DOC SIMILARITY --------------- #
            for doc in docs:

                # ----------- GET DOC VEC ----------------- #
                if doc.doc_id in self.doc_id_to_idx:
                    doc_idx = self.doc_id_to_idx[doc.doc_id]
                    doc_vec = self.embeddings[doc_idx]

                    # --------------- Cosine Similarity -------------- #
                    similarity = float(np.dot(doc_vec, query_vec))
                    doc.score = similarity
                    reranked.append(doc)
                else:
                    reranked.append(doc)

            # ------------- SORT SCORES ------------------ #
            reranked.sort(key=lambda x: x.score, reverse=True)

            return reranked[:top_k]

        except Exception as e:
            print(f"Warning: Embedding rerank failed: {e}")
            return docs[:top_k]


    def score_batch_with_llm(self, query: str, distribution: Dict, docs: List[RetrievedDoc]) -> List[float]:

        # ------------------------------ System Prompt For LLM ---------------------------- #
        sys_prompt = """
                    You are tasked with scoring documents for survey question analysis.

                    OBJECTIVE: Select documents that are DIVERSE yet PROPORTIONALLY REPRESENTATIVE of the overall opinion distribution.

                    Score each document with float values (0-1) based on:
                    - Does it contain OPINIONS about the survey topic?
                    - Does it express sentiment toward the answer options?
                    - Is it from real people discussing this topic?
                    - How clearly does it represent a specific viewpoint?

                    CRITICAL SCORING STRATEGY:

                    1. Identify Opinion Distribution First:
                    - Count how many documents support each answer option
                    - Determine the majority and minority opinions

                    2. Score Proportionally:
                    - If 70% of documents support Option A and 30% support Option B:
                        * High-quality Option A documents should score 0.7-0.9
                        * High-quality Option B documents should score 0.5-0.7
                    - This ensures top-ranked documents reflect the actual distribution

                    3. Guarantee Diversity:
                    - ALWAYS include at least one document per viewpoint if it exists
                    - Even minority opinions (with just 1-2 documents) should have their best representative scored 0.4-0.6
                    - DO NOT score all minority opinion documents below 0.3 or they won't be selected

                    4. Quality Tiers:
                    - Majority opinion + clear/strong: 0.75-0.95
                    - Majority opinion + moderate: 0.60-0.75
                    - Minority opinion + clear/strong: 0.50-0.70
                    - Minority opinion + moderate: 0.35-0.50
                    - Weak/unclear opinions: 0.10-0.30
                    - Off-topic/factual only: 0.0-0.10

                    PRIORITIZE documents with:
                    - Clear opinions/positions on the topic
                    - Discussion of the answer options
                    - Personal experiences related to the question
                    - Representative of their viewpoint category

                    DEPRIORITIZE documents that are:
                    - Purely factual/informational
                    - News articles without opinions
                    - Off-topic or tangentially related
                    - Duplicate perspectives already well-represented

                    Return JSON with precise scores (6-10 decimals): {"0": 0.847293, "1": 0.312456, "2": 0.923847, ...}
                    """

        # ------------------------ Build Descriptions for LLM ----------------------- #
        descs = []

        for id, doc in enumerate(docs):
            desc = f"Document {id}\n"

            if doc.title:
                desc += f"Title: {doc.title[:100]}\n"
            if doc.description:
                desc += f"Description: {doc.description[:100]}\n"
            if doc.content:
                desc += f"Content: {doc.content[:200]}\n"

            descs.append(desc)

        # ---------------------- BUILD CONTEXT ----------------------------- #
        usr_prompt = f"""
                        Survey Question: {query}

                        Answer options: {distribution}

                        Documents to score:
                        {"\n".join(descs)}

                        Score each document's relevance to the survey question (0-1).
                        """

        try:
            # ------------- Query LLM ---------- #
            result = self.client.chat_json(
                system=sys_prompt,
                user=usr_prompt,
                max_tokens=200,
                temperature=0.0
            )

            # ---------------- Extract scores ---------------------- #
            scores = []

            for index in range(len(docs)):
                score = result.get(str(index), 0.5)
                scores.append(float(score))

            return scores

        except:
            print("Warning: Failed to score batch...")
            return [0.5] * len(docs)


    def llm_rerank(
        self,
        query: str,
        distribution: Dict,
        docs: List[RetrievedDoc],
        top_k: int,
        batch_size: int = 15
    ) -> List[RetrievedDoc]:

        scored = []

        # ---------------- Batch process documents ---------------------- #
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i + batch_size]

            # --------------- score batch --------------- #
            batch_scores = self.score_batch_with_llm(query, distribution, batch)

            for doc, score in zip(batch, batch_scores):
                scored.append((doc, score))

        # ---------------- Sort by score ---------------------- #
        scored.sort(key=lambda x: x[1], reverse=True)

        # ---------------- Update scores and return top-k ---------------------- #
        reranked = []
        for doc, score in scored[:top_k]:
            doc.score = score
            reranked.append(doc)

        return reranked


    # SPARSE SEARCH -------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def sparse_search(self, query: str, top_k: int = 10) -> List[RetrievedDoc]:

        # --------- Check if Whoosh is available ----------- #
        if not self.sparse_index:
            return []

        results = []

        try:
            with self.sparse_index.searcher() as searcher:

                # ------ OR group + cleaned query = better recall on natural-language questions ------ #
                parser = MultifieldParser(
                    ["title", "description", "post_content", "content"],
                    schema=self.sparse_index.schema,
                    group=OrGroup,
                )

                q = parser.parse(_clean_query(query))
                hits = searcher.search(q, limit=top_k)

                # ---------------- Make RetrievedDoc and append ---------------------- #
                for hit in hits:
                    results.append(RetrievedDoc(
                        doc_id=hit["doc_id"],
                        title=hit.get("title", ""),
                        description=hit.get("description", ""),
                        post_content=hit.get("post_content", ""),
                        content=hit.get("content", ""),
                        score=hit.score,
                        source="sparse"
                    ))

        except:
            print("Warning: Whoosh search failed")

        return results


    # DENSE SEARCH -------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def dense_search(self, query: str, top_k: int = 10) -> List[RetrievedDoc]:

        # ---------------- Check embeddings available ---------------------- #
        if self.embeddings is None or self.doc_ids is None:
            return []
        if self.client is None:
            try:
                self.client = LLMClient()
            except:
                print("Warning: Failed to initialize client")
                return []

        try:
            # ---------------- Embed Query ---------------------- #
            q_vec = np.array(self.client.embed_one(query), dtype=np.float32)

            # ---------------- GET SIMILARITY  ---------------------- #
            similarity = np.dot(self.embeddings, q_vec)

            # ----------------------- SORT TOP_K ----------------- #
            top_k_ind = np.argsort(similarity)[::-1][:top_k]

            results = []

            # ---------------- Retrieve Full Content from Whoosh ---------------------- #
            if self.sparse_index:
                with self.sparse_index.searcher() as searcher:
                    for idx in top_k_ind:
                        doc_id = str(self.doc_ids[idx])
                        score = float(similarity[idx])

                        # -------------- GET DOC FIELDS ------------ #
                        doc_data = {
                            "title": "",
                            "description": "",
                            "post_content": "",
                            "content": ""
                        }

                        # ---------------- Extract Doc ID Match ---------------------- #
                        doc_q = Term("doc_id", doc_id)
                        hits = searcher.search(doc_q, limit=1)

                        if hits:
                            hit = hits[0]
                            doc_data["title"] = hit.get("title", "")
                            doc_data["description"] = hit.get("description", "")
                            doc_data["post_content"] = hit.get("post_content", "")
                            doc_data["content"] = hit.get("content", "")

                        results.append(RetrievedDoc(
                            doc_id=doc_id,
                            title=doc_data["title"],
                            content=doc_data["content"],
                            description=doc_data["description"],
                            post_content=doc_data["post_content"],
                            score=score,
                            source="dense"
                        ))

            else:
                # ------------------ NO DOC IN WHOOSH ------------ #
                for idx in top_k_ind:
                    results.append(RetrievedDoc(
                        doc_id=str(self.doc_ids[idx]),
                        title="",
                        content="",
                        description="",
                        post_content="",
                        score=float(similarity[idx]),
                        source="dense"
                    ))

            return results

        except Exception as e:
            print(f"Warning: Dense search failed: {type(e).__name__}: {e}")
            return []


    # HYBRID SEARCH -------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def hybrid_search(self, query: str, top_k: int = 10, alpha: float = 0.6) -> List[RetrievedDoc]:
        # Higher alpha = more sparse, lower alpha = more dense. RRF fusion.

        # ---------------- Get docs from sparse and dense ---------------------- #
        sparse_results = self.sparse_search(query, top_k=1500)
        dense_results = self.dense_search(query, top_k=1500)

        k = 60
        doc_scores = {}
        doc_map = {}

        for rank, doc in enumerate(sparse_results, 1):
            doc_scores[doc.doc_id] = alpha * (1.0 / (k + rank))
            doc_map[doc.doc_id] = doc

        for rank, doc in enumerate(dense_results, 1):
            score = (1 - alpha) * (1.0 / (k + rank))
            if doc.doc_id in doc_scores:
                doc_scores[doc.doc_id] += score
            else:
                doc_scores[doc.doc_id] = score
                doc_map[doc.doc_id] = doc

        sorted_ids = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)

        results = []
        for doc_id in sorted_ids[:top_k]:
            doc = doc_map[doc_id]
            doc.score = doc_scores[doc_id]
            results.append(doc)

        return results


    # MAIN RETRIEVAL -------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def retrieve(self, query: str, distribution: List = [], method: str = "hybrid", top_k: int = 10,
                 rerank: bool = False) -> List[RetrievedDoc]:

        # ---------------- Get results ---------------------- #
        if method == "sparse":
            docs = self.sparse_search(query, top_k * 3 if rerank else top_k)
        elif method == "dense":
            docs = self.dense_search(query, top_k * 3 if rerank else top_k)
        elif method == "hybrid":
            docs = self.hybrid_search(query, top_k * 3 if rerank else top_k)
        else:
            raise ValueError(f"Unknown retrieval method: {method}")

        if rerank:
            docs = self.llm_rerank(query, distribution, docs, top_k)

        return docs


    # GET SUPPORTS -------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def get_exactly_100_supports_v2(
        self,
        query: str,
        distribution: Dict,
        method: str = "hybrid"
    ) -> tuple[List[str], List[RetrievedDoc]]:

        all_docs = []
        seen = set()
        seen_lock = threading.Lock()

        print(f"  Phase 1: Collecting candidates...")

        # ------------------------ PHASE 1: GET 3k DOCS ---------------------------- #

        queries = self.expansion_agent.expand_query(query, distribution, max_variations=7)
        print(f"    Expanded to {len(queries)} queries")

        # ----------------------- DEFINE SEARCH FUNCTION -------------------------- #

        def search_one_query(exp_query):

            print(f"    Trying query: '{exp_query[:60]}...'")

            if method == "hybrid":
                retrieved_docs = self.hybrid_search(exp_query, top_k=1000)
            elif method == "dense":
                retrieved_docs = self.dense_search(exp_query, top_k=1000)
            elif method == "sparse":
                retrieved_docs = self.sparse_search(exp_query, top_k=1000)
            else:
                retrieved_docs = self.hybrid_search(exp_query, top_k=1000)

            # -------------- FILTER UNIQUE (THREAD SAFE) -------------- #
            unique_docs = []
            with seen_lock:
                for doc in retrieved_docs:
                    if doc.doc_id not in seen:
                        unique_docs.append(doc)
                        seen.add(doc.doc_id)

            return unique_docs

        # ----------------------- TRY QUERIES IN PARALLEL -------------------------- #

        print(f"    Executing {len(queries)} queries in parallel (max_workers=4)...")

        try:
            with ThreadPoolExecutor(max_workers=4) as executor:
                future_to_query = {executor.submit(search_one_query, q): q for q in queries}

                for future in as_completed(future_to_query):
                    try:
                        unique_docs = future.result()
                        all_docs.extend(unique_docs)

                        print(f"      Total unique docs so far: {len(all_docs)}")

                        if len(all_docs) >= 3000:
                            print(f"    Collected enough docs, stopping at {len(all_docs)}")
                            break

                    except Exception as e:
                        print(f"      ERROR: {e}")

        except Exception as e:
            print(f"    ERROR in parallel execution: {e}")
            print(f"    Falling back to sequential...")

            for exp_query in queries:
                try:
                    unique_docs = search_one_query(exp_query)
                    all_docs.extend(unique_docs)

                    print(f"      Total unique docs so far: {len(all_docs)}")

                    if len(all_docs) >= 3000:
                        break
                except:
                    continue

        print(f"  Phase 1 Complete: Collected {len(all_docs)} docs")

        if len(all_docs) == 0:
            print("  ERROR: No documents found!")
            return [], []

        # -------------------- PHASE 2: EMBEDDING RERANK -------------------------

        print(f"  Phase 2: Embedding rerank...")

        all_docs.sort(key=lambda x: x.score, reverse=True)

        top_docs = self.embedding_rerank(query, distribution, all_docs[:500], top_k=150)

        print(f"    Embedding reranked to {len(top_docs)} docs")

        # ------------------------ PHASE 3: LLM RERANK ------------------------------ #

        print(f"  Phase 3: LLM rerank...")

        top_docs = self.llm_rerank(query, distribution, top_docs[:30], top_k=15)

        # --------------------------- PHASE 4: BUILD FINAL 100 SUPPORT LIST ------------------------------ #

        print(f"  Phase 4: Building final support list...")

        support_docs = []
        support_seen = set()

        for doc in top_docs:
            if doc.doc_id not in support_seen:
                support_docs.append(doc)
                support_seen.add(doc.doc_id)

        if len(support_docs) < 100:
            for doc in all_docs:
                if doc.doc_id not in support_seen:
                    support_docs.append(doc)
                    support_seen.add(doc.doc_id)
                if len(support_docs) >= 100:
                    break

        # ------------------------ Fill 100 docs if not enough --------------------------------- #

        if len(support_docs) < 100 and self.doc_ids is not None:
            remaining = 100 - len(support_docs)
            rng = np.random.RandomState(42)
            available = [doc_id for doc_id in self.doc_ids if doc_id not in support_seen]

            if available:
                # --------------------- random padding if not enough docs --------------------------- #
                padding_ids = rng.choice(available, size=min(remaining, len(available)), replace=False)
                for pad_id in padding_ids:
                    support_docs.append(RetrievedDoc(
                        doc_id=pad_id,
                        title="",
                        content="",
                        score=0.0,
                        source="random_padding"
                    ))

        support_ids = [doc.doc_id for doc in support_docs[:100]]

        print(f"  Final support list: {len(support_ids)} doc IDs")

        return support_ids[:100], top_docs[:15]
