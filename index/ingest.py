from __future__ import annotations
import sys
import pathlib
from typing import List
from dataclasses import dataclass

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from utils.utils import load_docs


@dataclass
class Chunk:
    doc_id: str
    chunk_id: int
    title: str
    source: str
    description: str
    post_content: str
    content: str


def load_jsonl_chunk(path: str) -> List[Chunk]:

    # ------------ Load jsonl to list of Dict -------------- #
    list_docs = load_docs(path)

    out: List[Chunk] = []
    i = 0

    # --------------- LOAD DATA FROM DICT INTO CHUNKS --------------- #
    for doc_dict in list_docs:

        chunk = Chunk(
            doc_id=doc_dict.get("id", ""),
            chunk_id=i,
            title=doc_dict.get("title", ""),
            source=doc_dict.get("source", ""),
            description=doc_dict.get("description", ""),
            content=doc_dict.get("content", ""),
            post_content=doc_dict.get("post_content", ""),
        )

        out.append(chunk)
        i += 1

    return out


def load_text(chunk: Chunk) -> str:
    text = [chunk.title, chunk.description, chunk.post_content, chunk.content]
    return "\n".join([t for t in text]).strip()


def split_into_word_chunks(text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:

    # ------------ split text into tokens --------------- #
    words = text.split()

    if len(words) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk_text = ' '.join(words[start:end])
        chunks.append(chunk_text)

        start += (chunk_size - overlap)

        if end >= len(words):
            break

    return chunks


def split_chunk_into_subchunks(chunk: Chunk, chunk_size: int = 200, overlap: int = 50) -> List[Chunk]:

    # ------------------------ GET TEXT AND SPLIT ------------------------- #
    full_text = load_text(chunk)
    text_chunks = split_into_word_chunks(full_text, chunk_size=chunk_size, overlap=overlap)

    # ---------------------------------- CREATE SUBCHUNKS --------------------------------- #
    sub_chunks = []
    for i, text_chunk in enumerate(text_chunks):
        sub_chunk = Chunk(
            doc_id=f"{chunk.doc_id}",
            chunk_id=i,
            title=chunk.title,
            source=chunk.source,
            description=chunk.description if i == 0 else "",
            post_content="",
            content=text_chunk
        )
        sub_chunks.append(sub_chunk)

    return sub_chunks


def load_jsonl_with_subchunking(path: str, sub_chunk: bool = False, chunk_size: int = 300, overlap: int = 25) -> List[Chunk]:

    # ------------------ CHUNK TEXT ----------------------------- #
    original_chunks = load_jsonl_chunk(path)

    if not sub_chunk:
        return original_chunks

    # ---------------------- SPLIT CHUNK TEXT ----------------------------------------------- #
    all_subchunks = []
    for chunk in original_chunks:
        sub_chunks = split_chunk_into_subchunks(chunk, chunk_size=chunk_size, overlap=overlap)
        all_subchunks.extend(sub_chunks)

    print(f"Original documents: {len(original_chunks)}")
    print(f"Total sub-chunks: {len(all_subchunks)}")
    print(f"Average sub-chunks per document: {len(all_subchunks) / len(original_chunks):.2f}")

    return all_subchunks
