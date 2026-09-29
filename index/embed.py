from __future__ import annotations
import os, sys, pathlib, yaml, argparse, json
import numpy as np
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from utils.llm_client import LLMClient
from index.ingest import load_jsonl_with_subchunking, load_text


BATCH_SIZE   = 1000     # docs per API call (tune vs token cap)
MAX_CHARS    = 6000    # ~1500 tokens per doc; keeps batch under 300k tok cap
CKPT_EVERY   = 20      # flush checkpoint every N batches


def embed_docs(args: argparse.Namespace):

    DATA_DIR = "data"

    # -------------- Load config ------------------- #
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    embedding_dir  = pathlib.Path(config["index"]["emb_dir"])
    embedding_dir.mkdir(parents=True, exist_ok=True)
    documents_path = config[DATA_DIR]["documents_path"]

    ckpt_path = embedding_dir / "embed_checkpoint.jsonl"
    out_path  = embedding_dir / "id_to_embedding.npz"

    # ----------- Load docs --------------- #
    docs = load_jsonl_with_subchunking(documents_path)
    print(f"Loaded {len(docs)} docs")

    ids       = [chnk.doc_id for chnk in docs]
    text_list = [load_text(chnk)[:MAX_CHARS] for chnk in docs]

    # ----------- Resume from checkpoint ---------- #
    done_ids: dict[str, list[float]] = {}
    if ckpt_path.exists():
        print(f"Resuming from {ckpt_path}")
        with open(ckpt_path) as f:
            for line in f:
                rec = json.loads(line)
                done_ids[rec["id"]] = rec["vec"]
        print(f"  {len(done_ids)} already embedded, skipping")

    # ----------- Embed --------------- #
    client = LLMClient()

    if os.environ.get("EMBED_YES") != "1":
        if input("Begin embedding? Yes/No: ").strip().lower() != "yes":
            print("Aborted.")
            return

    # ------------------- Embed in batches with checkpoint ------------------- #
    print(f"Embedding with {client.embed_model} (batch={BATCH_SIZE}, max_chars={MAX_CHARS})...")

    # ----------- filter out already-done docs ------------ #
    todo = [(i, ids[i], text_list[i]) for i in range(len(ids)) if ids[i] not in done_ids]
    print(f"  {len(todo)} docs remaining")

    t_start = time.time()
    batches_since_ckpt = 0

    with open(ckpt_path, "a") as ckpt_f:

        for batch_start in range(0, len(todo), BATCH_SIZE):
            batch     = todo[batch_start:batch_start + BATCH_SIZE]
            batch_ids = [b[1] for b in batch]
            batch_txt = [b[2] for b in batch]

            try:
                vecs = client.embed_batch(batch_txt)
            except Exception as e:
                print(f"  [batch fail @ {batch_start}] {type(e).__name__}: {str(e)[:200]}")
                # -------- fall back to per-doc so bad doc doesn't kill the batch ------- #
                vecs = []
                for txt in batch_txt:
                    try:
                        vecs.append(client.embed_batch([txt])[0])
                    except Exception as e2:
                        print(f"    skip: {type(e2).__name__}")
                        vecs.append(None)

            # ---------- write each result to checkpoint immediately ---------- #
            for did, v in zip(batch_ids, vecs):
                if v is None:
                    continue
                done_ids[did] = v
                ckpt_f.write(json.dumps({"id": did, "vec": v}) + "\n")

            batches_since_ckpt += 1
            if batches_since_ckpt >= CKPT_EVERY:
                ckpt_f.flush()
                batches_since_ckpt = 0

            done  = len(done_ids)
            rate  = (done - (len(ids) - len(todo))) / max(1e-6, time.time() - t_start)
            eta_s = (len(ids) - done) / max(1e-6, rate) if rate > 0 else 0
            print(f"  {done}/{len(ids)}  ({rate:.0f} docs/s, ETA {eta_s/60:.1f} min)")

    print(f"Done. {len(done_ids)}/{len(ids)} embedded.")

    # ---------------------- Convert checkpoint → final .npz ---------------------- #
    print(f"Writing {out_path}...")

    ordered_ids = [i for i in ids if i in done_ids]
    ordered_vec = [done_ids[i] for i in ordered_ids]

    np.savez_compressed(
        out_path,
        model      = np.array(client.embed_model),
        created_at = np.array(int(time.time())),
        ids        = np.array(ordered_ids, dtype=object),
        embeddings = np.array(ordered_vec, dtype=np.float32),
    )
    print(f"Saved {len(ordered_vec)} embeddings")
