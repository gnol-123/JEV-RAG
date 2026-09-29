from __future__ import annotations
import os, json, time, math, re
from typing import List, Dict, Any, Optional, Sequence
from dataclasses import dataclass

try:
    from retry.api import retry_call
except Exception:
    retry_call = None

from together import Together

_ST_MODEL = None   # lazy singleton — load model once per process


# EMBED_BACKEND env var picks provider:
#   "cuda" (default) → local sentence-transformers, GPU if available
#   "api"            → OpenAI embeddings

def _embed_defaults():
    backend = os.environ.get("EMBED_BACKEND", "cuda").lower()
    if backend == "api":
        return "text-embedding-3-small"
    return "BAAI/bge-base-en-v1.5"


@dataclass
class LLMClient:
    api_key: str     = os.environ.get("TOGETHER_API_KEY", "")
    chat_model: str  = os.environ.get("FACTCHECK_CHAT_MODEL", "meta-llama/Llama-3.3-70B-Instruct-Turbo")
    embed_model: str = os.environ.get("FACTCHECK_EMBED_MODEL") or _embed_defaults()
    request_timeout_s: int = 60

    def __post_init__(self):
        if not self.api_key:
            raise RuntimeError("TOGETHER_API_KEY is not set.")
        self.client        = Together(api_key=self.api_key)
        self.embed_backend = os.environ.get("EMBED_BACKEND", "cuda").lower()
        self.openai_client = None
        if self.embed_backend == "api":
            from openai import OpenAI
            key = os.environ.get("OPENAI_API_KEY", "")
            if not key:
                raise RuntimeError("EMBED_BACKEND=api but OPENAI_API_KEY not set.")
            self.openai_client = OpenAI(api_key=key)

    # ---------- Embeddings ----------

    def _get_st_model(self):
        global _ST_MODEL
        if _ST_MODEL is None:
            from sentence_transformers import SentenceTransformer
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[LLMClient] Loading {self.embed_model} on {device}...")
            _ST_MODEL = SentenceTransformer(self.embed_model, device=device)
        return _ST_MODEL

    def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        # ---------- API path (OpenAI) ---------- #
        if self.embed_backend == "api":
            def call():
                return self.openai_client.embeddings.create(model=self.embed_model, input=list(texts))
            resp = self._with_retry(call)
            vecs = [d.embedding for d in resp.data]
            out = []
            for v in vecs:
                norm = math.sqrt(sum(x * x for x in v)) or 1.0
                out.append([x / norm for x in v])
            return out

        # ---------- Local path (sentence-transformers, GPU) ---------- #
        model = self._get_st_model()
        vecs = model.encode(
            list(texts),
            batch_size=128,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [v.tolist() for v in vecs]

    def embed_one(self, text: str) -> List[float]:
        vecs = self.embed_batch([text])
        return vecs[0] if vecs else []

    async def embed_one_async(self, text: str) -> List[float]:
        vecs = await self.embed_many_async([text], batch_size=1, concurrency=1)
        return vecs[0] if vecs else []

    async def embed_many_async(self, texts: Sequence[str], batch_size: int = 16, concurrency: int = 4) -> List[List[float]]:
        import asyncio
        from asyncio import Semaphore

        if not texts:
            return []

        batches = [(i, list(texts[i:i + batch_size])) for i in range(0, len(texts), batch_size)]
        results: List[Optional[List[float]]] = [None] * len(texts)
        sem = Semaphore(max(1, concurrency))

        async def worker(start_index: int, batch_inputs: Sequence[str]) -> None:
            async with sem:
                vecs = await asyncio.to_thread(self.embed_batch, batch_inputs)
                for offset, v in enumerate(vecs):
                    results[start_index + offset] = v

        await asyncio.gather(*[worker(start, batch) for (start, batch) in batches])
        return [v for v in results if v is not None]

    # ---------- Chat ----------

    def chat(self, messages, *, max_tokens=600, temperature=0.2, extra=None) -> str:
        def call():
            return self.client.chat.completions.create(
                model=self.chat_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **(extra or {}),
            )
        resp = self._with_retry(call)
        return resp.choices[0].message.content

    async def chat_async(self, messages, *, max_tokens=400, temperature=0.2, extra=None) -> str:
        import asyncio
        return await asyncio.to_thread(self.chat, messages, max_tokens=max_tokens, temperature=temperature, extra=extra)

    # ---------- Chat JSON ----------

    def chat_json(self, system: str, user: str, *, schema_hint: Optional[str] = None,
                  max_tokens: int = 600, temperature: float = 0.0) -> Dict[str, Any]:
        sys_msg = system.strip()
        if schema_hint:
            sys_msg += "\n\nReturn ONLY valid JSON. JSON schema (hint):\n" + schema_hint

        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user},
        ]
        extra = {"response_format": {"type": "json_object"}}
        txt = self.chat(messages, max_tokens=max_tokens, temperature=temperature, extra=extra)

        try:
            return json.loads(txt)
        except Exception:
            m = re.search(r"\{(?:.|\n)*\}", txt)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    pass
        return {"_error": "invalid_json_from_model", "_raw": txt}

    async def chat_json_async(self, system: str, user: str, *, schema_hint: Optional[str] = None,
                              max_tokens: int = 400, temperature: float = 0.0) -> Dict[str, Any]:
        import asyncio
        return await asyncio.to_thread(self.chat_json, system, user, schema_hint=schema_hint, max_tokens=max_tokens, temperature=temperature)

    # ---------- Retry ----------

    def _with_retry(self, fn):
        def _log(e, attempt):
            print(f"  [retry attempt {attempt}] {type(e).__name__}: {str(e)[:200]}")

        if retry_call is not None:
            # wrap so we log each retry
            def wrapped():
                try:
                    return fn()
                except Exception as e:
                    _log(e, "?")
                    raise
            return retry_call(wrapped, tries=4, delay=1, backoff=2, max_delay=10, exceptions=Exception, logger=None)

        delay = 1.0
        for attempt in range(4):
            try:
                return fn()
            except Exception as e:
                _log(e, attempt + 1)
                if attempt == 3:
                    raise
                time.sleep(delay)
                delay = min(10.0, delay * 2.0)
