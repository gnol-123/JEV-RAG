import argparse, sys, pathlib, os

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from index.whoosh_index import build_index


def _load_env(path: str = ".env"):
    if not pathlib.Path(path).exists():
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build sparse index + optional embeddings")
    p.add_argument("--config", type=str, default="config.yaml")

    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--embed-cuda", action="store_true",
                     help="After sparse build, embed docs locally (sentence-transformers + GPU)")
    grp.add_argument("--embed-api",  action="store_true",
                     help="After sparse build, embed docs via OpenAI API")

    return p.parse_args()


def main() -> None:
    _load_env()
    args = parse_args()

    if not os.environ.get("TOGETHER_API_KEY"):
        print("ERROR: TOGETHER_API_KEY not set. Add it to .env")
        sys.exit(1)

    print(f"Config: {args.config}")

    # ------------ Sparse index (always) ------------ #
    print("Building sparse index...")
    build_index(args)
    print("Sparse index done.")

    # ------------ Embed (opt-in) ------------ #
    if not (args.embed_cuda or args.embed_api):
        print("Skipping embed. Pass --embed-cuda or --embed-api to embed.")
        return

    if args.embed_api:
        os.environ["EMBED_BACKEND"] = "api"
        if not os.environ.get("OPENAI_API_KEY"):
            print("ERROR: --embed-api requires OPENAI_API_KEY in .env")
            sys.exit(1)
        print("Embedding via OpenAI API...")
    else:
        os.environ["EMBED_BACKEND"] = "cuda"
        print("Embedding locally on GPU/CPU...")

    # ------------ Import AFTER env is set (embed.py picks up backend) ------------ #
    from index.embed import embed_docs
    embed_docs(args)
    print("Embeddings done.")


if __name__ == "__main__":
    main()
