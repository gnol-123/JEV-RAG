import sys, pathlib, argparse, yaml

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from utils.query_utils import load_query_json


def load_qeury_from_arg(args: argparse.Namespace):

    DATA_DIR = "data"

    # ------------ Open config ----------------- #
    try:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

    # ----------- Get question path ---------- #
    query_path = config[DATA_DIR]["questions_path"]

    return load_query_json(query_path)
