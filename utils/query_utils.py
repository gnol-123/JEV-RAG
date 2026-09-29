import sys, pathlib
from dataclasses import dataclass

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from utils.utils import load_query


# ------------ QUERY DATA CLASS ------------ #

@dataclass
class query:
    qid: int
    query: str
    distribution: dict[str, int]
    support: list


# -------------- LOAD QUERY INTO QUERY DATA CLASS ------------ #

def load_query_json(path: str):
    all_query_dict = load_query(path)

    out: list[query] = []
    i = 0

    for query_str, query_data in all_query_dict.items():
        qid = i
        curr_query = query_str
        distribution = query_data["distribution"]
        support = query_data.get("supports", [])

        qry = query(
            qid=qid,
            query=curr_query,
            distribution=distribution,
            support=support
        )

        out.append(qry)
        i += 1

    return out
