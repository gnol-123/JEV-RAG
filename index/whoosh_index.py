import os, sys, pathlib, yaml, shutil, argparse
from whoosh.fields import Schema, ID, TEXT, STORED
from whoosh.analysis import StemmingAnalyzer
from whoosh import index

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from index.ingest import load_jsonl_with_subchunking


schema = Schema(
    doc_id=ID(stored=True),
    chunk_id=ID(stored=True),
    title=TEXT(stored=True, analyzer=StemmingAnalyzer()),
    description=TEXT(stored=True, analyzer=StemmingAnalyzer()),
    post_content=TEXT(stored=True, analyzer=StemmingAnalyzer()),
    content=TEXT(stored=True, analyzer=StemmingAnalyzer()),
    chunk_index=STORED,
    total_chunks=STORED
)


def build_index(args: argparse.Namespace):

    INDEX_DIR = "index"
    DATA_DIR  = "data"

    # ------- Load config --------------- #
    try:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

    # ----------------- OUTPUT DIR ------------- #
    index_dir = config[INDEX_DIR]["output_dir"]

    if os.path.exists(index_dir):
        print(f"Removing existing index at {index_dir}")
        shutil.rmtree(index_dir)

    os.makedirs(index_dir, exist_ok=True)
    print(f"Creating new index at {index_dir}")

    # ---------------- Create index ---------------- #
    indx = index.create_in(index_dir, schema)

    # ---------------- Load docs ---------------- #
    doc_dir = config[DATA_DIR]["documents_path"]
    print(f"Loading documents from {doc_dir}")
    docs = load_jsonl_with_subchunking(doc_dir, False)
    print(f"Loaded {len(docs)} document chunks")

    # ------------------- Whoosh writer -------------------- #
    writer = indx.writer(limitmb=256, procs=1, multisegment=False)

    successful  = 0
    failed      = 0
    empty_content = 0
    doc_ids_seen  = set()

    # ------------------------ FILL INDEX ----------------------- #
    for i, ch in enumerate(docs):
        try:
            doc_id       = ch.doc_id if ch.doc_id else f"unknown_{i}"
            title        = ch.title if ch.title else ""
            description  = ch.description if ch.description else ""
            post_content = ch.post_content if ch.post_content else ""
            content      = ch.content if ch.content else ""

            if not content.strip() and not title.strip():
                empty_content += 1
                print(f"Warning: Skipping chunk {i} (doc_id: {doc_id}) - no content")
                continue

            if "_chunk_" in doc_id:
                chunk_id    = doc_id
                base_doc_id = doc_id.split("_chunk_")[0]
            else:
                chunk_id    = f"{doc_id}_chunk_0"
                base_doc_id = doc_id

            doc_ids_seen.add(base_doc_id)

            # ------------- ADD DOC TO WRITER ------------- #
            writer.add_document(
                doc_id=base_doc_id,
                chunk_id=chunk_id,
                title=title,
                description=description,
                post_content=post_content,
                content=content,
                chunk_index=i,
                total_chunks=len(docs)
            )

            successful += 1

            if (i + 1) % 1000 == 0:
                print(f"Indexed {i + 1}/{len(docs)} chunks...")

        except Exception as e:
            failed += 1
            print(f"Error indexing chunk {i} (doc_id: {ch.doc_id}): {e}")

    # ----------- Commit -------------- #
    print("Writing to index...")
    writer.commit()

    print("INDEX BUILD COMPLETE " + "-" * 50)
    print(f"Total chunks processed: {len(docs)}")
    print(f"Successfully indexed:   {successful}")
    print(f"Failed to index:        {failed}")
    print(f"Skipped (empty):        {empty_content}")
    print(f"Unique documents:       {len(doc_ids_seen)}")
