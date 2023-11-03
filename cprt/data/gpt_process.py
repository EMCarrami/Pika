import os
import pickle
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob
from typing import Any, Dict, List

import openai
from tqdm import tqdm

from cprt.utils import DATA_PATH

INSTRUCTIONS = """You will receive details about a specific protein. Perform the following tasks:
1) Provide a factual summary, with a maximum of 500 words, that accurately and scientifically describes the functional,
biochemical and structural properties of this protein based only on the provided information.
Ensure that the summary follows a natural and scientific flow, starting with general information such as structure,
localization and taxonomy before detailing functional and biochemical properties.
Ensure that all key points are covered and DON'T provide any further information than what is stated in the input.
2) For each type of information provided, create a question-and-answer pair to elucidate an aspect of
the protein's functionality or biochemical properties without using the protein's name.

Use the below ; separated two-element tuples structure for the output:
(summary: this protein...); (what is question 1?, the answer 1); (what is question 2?, the answer 2); ...

For both tasks if the input contains large group of properties only provide the canonical and crucial information
rather than enumerating every single entry.
DON'T use any of your knowledge to add additional context or information. DON'T add any speculative or unwarranted
information that is not specifically provided.
AVOID using generic phrases or embellished terms such as 'highly', 'several', 'diverse' and 'various'.
"""


def fetch(payload: Dict[str, Any]) -> None:
    """Fetch function for ChatGPT API."""
    if not os.path.exists(f"{DATA_PATH}/gpt/{payload['uniprot_id']}_gpt.pkl"):
        try:
            print(f"submitted {payload['uniprot_id']} to gpt-3.5-turbo\n")
            response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=payload["message"])
        except openai.error.InvalidRequestError:
            print(f"submitted {payload['uniprot_id']} to gpt-3.5-turbo-16k\n")
            response = openai.ChatCompletion.create(model="gpt-3.5-turbo-16k", messages=payload["message"])
        except Exception as e:
            print(f"Error for {payload['uniprot_id']}: {e}")
            raise Exception(f"Error for {payload['uniprot_id']}: {e}")

        print(f"completed {payload['uniprot_id']}")
        with open(f"{DATA_PATH}/gpt/{payload['uniprot_id']}_gpt.pkl", "wb") as f:
            pickle.dump(response.to_dict(), f)


def run_gpt() -> None:
    """Run parallel requests to ChatGPT."""
    openai.api_key = os.environ["OPENAI_API_KEY"]
    with open(f"{DATA_PATH}/uniref50_subsample_data.pkl", "rb") as f:
        uniref_dict = pickle.load(f)
    tasks = []
    for uid, t in uniref_dict.items():
        tasks.append(
            {
                "uniprot_id": uid,
                "message": [
                    {"role": "system", "content": "You are a helpful assistant following all instructions exactly."},
                    {"role": "user", "content": INSTRUCTIONS},
                    {"role": "user", "content": str({k: v for k, v in t.items() if k != "sequence"})},
                ],
            }
        )
    random.shuffle(tasks)
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(fetch, task) for task in tasks[:10000]]
        print(f"Submitted {len(futures)} tasks")
        for future in as_completed(futures):
            exc = future.exception()
            if exc is not None:
                for f in futures:  # type: ignore[assignment]
                    f.cancel()  # type: ignore[attr-defined]
                raise Exception from exc


def merge_gtp_results() -> Dict[str, Dict[str, List[str] | str]]:
    """Merge results from chatGPT."""
    out = {}
    with open(f"{DATA_PATH}/uniref50_subsample_data.pkl", "rb") as f:
        uniref_dict = pickle.load(f)
    fls = glob(f"{DATA_PATH}/gpt/*.pkl")
    for idx, fl in tqdm(enumerate(fls)):
        uid = fl.split("/")[-1].split("_")[0]
        seq = uniref_dict[uid]["sequence"]
        # _data = uniref_dict[uid]["protein size"]
        # _aa = _data[0].replace(" aa", "")
        # _kda = _data[1].replace(" KDa", "")
        with open(fl, "rb") as f:
            data_load = pickle.load(f)["choices"][0]["message"]["content"]
            data_load = data_load.replace(" ;", ";").replace(" .", ".").replace("?;", "?,").replace(");", ")\n")
        data = []
        new_entry = ""
        for d in data_load.split("\n"):
            _d = d.strip().strip(";").strip()
            if "?" in _d:
                data.append(clean_entry(new_entry))
                new_entry = _d
            elif len(_d) > 1:
                new_entry += f" {_d}"
        data.append(clean_entry(new_entry))
        try:
            assert len(data) > 1 or "?" not in " ".join(data), "split didnt work"
            assert all(["?" in i for i in data[1:]]) or "?" not in " ".join(data), "missing Q"
            # assert _aa in data_load.replace(',', '') or _kda in data_load.replace(',', ''), 'wrong size'
            out[uid] = {"info": data, "sequence": seq}
        except Exception:
            raise ValueError(f"{idx} {fl} \n {data}")
    return out


def clean_entry(txt: str) -> str:
    """Clean entry results from ChaGPT."""
    txt = txt.strip().replace("?,", "?").replace(").", ")")
    if txt.startswith("("):
        inner = 0
        for i, s in enumerate(txt):
            if s == ")":
                if inner == 1:
                    txt = txt[1:i] + txt[i + 1 :]
                    break
                else:
                    inner -= 1
            if s == "(":
                inner += 1
    return txt.strip()


if __name__ == "__main__":
    import time

    s = time.time()
    run_gpt()
    print(time.time() - s)
    merge_gtp_results()
