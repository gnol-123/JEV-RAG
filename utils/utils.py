import json, sys
from typing import List


def load_docs(path):
    raw = []
    seen_ids = set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Skipping invalid JSON line: {e}")
                    continue

                doc_id = obj.get("id")
                if doc_id:
                    if doc_id in seen_ids:
                        continue
                    seen_ids.add(doc_id)

                raw.append(obj)

    except Exception as e:
        print(f"Error loading documents: {e}")
        sys.exit(1)

    return raw


def load_query(path: str):
    raw = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as e:
        print(f"Error loading queries: {e}")
        sys.exit(1)
    return raw
