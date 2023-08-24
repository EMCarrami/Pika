import csv
import gzip
import pickle
from collections import defaultdict
from copy import deepcopy
from typing import Dict, List, Literal, Tuple, Union, cast

import numpy as np
import pandas as pd
from tqdm import tqdm

from cprt.utils import DATA_PATH

BASE_FIELDS = ["sequence", "taxon", "mw", "length"]

INFO_FIELDS = [
    "GO",
    "Pfam",
    "InterPro",
    "Gene3D",
    "SUPFAM",
    "catalytic activity",
    "cofactor",
    "subunit",
    "subcellular location",
    "pH dependence",
    "temperature dependence",
]

# TODO: Include helix and strands in data fields


def process_id_mapping() -> None:
    """
    Process id mapping file from uniprot.

    Creates dict mappings from uniref clusters to all uniprot ids >= 30aa and with at least 1 info field.
    swiss_prot ids downloaded from uniprot with name and length columns: uniprotkb_reviewed_true_2023_08_14.tsv
    idmapping file downloaded from uniprot ftp: idmapping_selected.tab.gz
    files were processed to remove unnecessary data:
    awk '{
        if (NR==FNR) {if ($3>29) A[$1];}
        else {
            if ($1 in A) print $0
            }
        }' uniprotkb_reviewed_true_2023_08_14.tsv <(zcat idmapping_selected.tab.gz) > uniprot_to_uniref.tsv
    """
    # TODO: change the pre-pre-process to use yield
    id_map = pd.read_csv(f"{DATA_PATH}/uniprot_to_uniref.tsv", sep="\t", header=None)[
        [0, 7, 8, 9]
    ]
    id_map.rename(
        columns={0: "id", 7: "uniref100", 8: "uniref90", 9: "uniref50"}, inplace=True
    )
    uniref90_dict, uniref50_dict = defaultdict(list), defaultdict(list)
    for _, row in tqdm(id_map.iterrows(), total=len(id_map)):
        if get_num_info_fields(row.id) > 0:
            uniref50_dict[row.uniref50].append(row.id)
            uniref90_dict[row.uniref90].append(row.id)

    merged50 = merge_clusters(uniref50_dict)
    merged90 = merge_clusters(uniref90_dict)

    dicts = [uniref50_dict, uniref90_dict, merged50, merged90]
    file_names = [
        "uniref50_dict",
        "uniref90_dict",
        "uniref50_merged_dict",
        "uniref90_merged_dict",
    ]
    for dct, fn in zip(dicts, file_names):
        with open(f"{DATA_PATH}/{fn}.pkl", "wb") as f:
            pickle.dump(dct, f)


def get_num_info_fields(uniprot_id: str) -> int:
    """Get number of useful info fields for the given uniprot_id: n from pkl file."""
    with open(f"{DATA_PATH}/swiss_prot/{uniprot_id}.pkl", "rb") as f:
        d = pickle.load(f)
    return len([k for k in d if k in INFO_FIELDS])


def merge_clusters(uniref_dict: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Merge uniref clusters to reduce isoform-clusters."""
    new_dict = deepcopy(uniref_dict)
    # move all proteins clustered in an isoform-named-cluster to the main cluster of that isoform
    to_del = []
    for k, v in new_dict.items():
        if "-" in k:
            # k = Unirefxx_yyyyy-n
            base_cluster = k.split("-")[0]
            if base_cluster in new_dict:
                new_dict[base_cluster] += v
                to_del.append(k)
            else:
                # if main protein not a cluster, move it there if that protein is a member of another cluster.
                for _k, _v in new_dict.items():
                    # base_cluster = Unirefxx_yyyyy
                    if (
                        base_cluster.split("_")[1] in _v
                        and base_cluster not in _k
                        and _k not in to_del
                    ):
                        new_dict[_k] += v
                        to_del.append(k)
                        break
    for k in to_del:
        del new_dict[k]

    new_size, original_size = 0, 0
    for v in uniref_dict.values():
        original_size += len(v)
    for v in new_dict.values():
        new_size += len(v)
    assert new_size == original_size

    return new_dict


def subsample_clusters_by_gzip_score(uniref_identity: Literal["50", "90"]) -> None:
    """Subsample uniref custer to maximum of two high-information representatives."""
    with open(f"{DATA_PATH}/uniref{uniref_identity}_merged_dict.pkl", "rb") as f:
        uniref_dict = pickle.load(f)

    out_file_name = f"{DATA_PATH}/uniref{uniref_identity}_gzip_subsample.csv"
    with open(out_file_name, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("cluster_id", "uniprot_id", "info_fields", "rank_in_cluster"))

    dataset = {}
    for uniref, uniprots in tqdm(uniref_dict.items()):
        # get data for valid uniprot_ids in cluster
        df, cluster_rep_uid = get_uniref_cluster_data(uniprots, uniref)

        id_list = df["uniprot_id"].to_list()
        data_dicts = df["data_dict"].to_list()
        info_fields = df["data_fields"].to_list()
        text_data = df["text_data"].to_list()
        gzip_info = df["gzip_info"].to_list()

        if len(id_list) > 1:
            top_ranks = get_top_info_ids(id_list, text_data, gzip_info, cluster_rep_uid)
            for r, idx in enumerate(top_ranks):
                uid = id_list[idx]
                dataset[uid] = data_dicts[idx]
                add_line_to_csv((uniref, uid, info_fields[idx], r + 1), out_file_name)
        else:
            dataset[id_list[0]] = data_dicts[0]
            add_line_to_csv((uniref, id_list[0], info_fields[0], 0), out_file_name)

    with open(f"{DATA_PATH}/uniref90_subsample_data.pkl", "wb") as f:
        pickle.dump(dataset, f)


def get_uniref_cluster_data(
    id_list: List[str], uniref_id: str
) -> Tuple[pd.DataFrame, str]:
    """
    Collect and filter the data for each uniprot_id.

    Filtering based on ratio of the sequence length to the median of the cluster
    :param id_list: list of all uniprot_ids in the uniref cluster
    :param uniref_id: uniref cluster id

    :return: filtered DataFrame with columns uniprot_id, data_dict, data_fields, text_data, gzip_info
            and uid of cluster_representative if among uniprot_ids, otherwise ''
    """
    # data_rows: uniprot_id, data_dict, data_fields, text_data, gzip_info
    data_rows = []
    lengths = []
    cluster_rep_uid = ""

    for uid in id_list:
        cluster_rep_uid = uid if uid in uniref_id else cluster_rep_uid
        with open(f"{DATA_PATH}/swiss_prot/{uid}.pkl", "rb") as f:
            info_dict = pickle.load(f)

        lengths.append(info_dict["length"])

        # text of info_fields for gzip info
        info_fields = [k for k in info_dict if k in INFO_FIELDS]
        text_data = "\n".join([f'{k}: {"; ".join(info_dict[k])}' for k in info_fields])

        # all fields for dataset
        all_fields = BASE_FIELDS + info_fields

        data_rows.append(
            (
                uid,
                {k: info_dict[k] for k in all_fields},
                ";".join(all_fields),
                text_data,
                len(gzip.compress(text_data.encode())),
            )
        )

    median_len: float = np.median(lengths)
    len_ratios = np.array(lengths) / median_len

    df = pd.DataFrame(
        data_rows,
        columns=["uniprot_id", "data_dict", "data_fields", "text_data", "gzip_info"],
    )
    df = df[len_ratios > 0.25]
    df.reset_index(drop=True, inplace=True)

    return df, cluster_rep_uid


def get_top_info_ids(
    id_list: List[str], text_data: List[str], gzip_info: List[int], cluster_rep_uid: str
) -> Tuple[int, int]:
    """Find the top ranking indices of data with highest gzip info."""
    cluster_rep_idx = (
        id_list.index(cluster_rep_uid) if cluster_rep_uid in id_list else None
    )

    rank1_idx = get_best_score_idx(gzip_info, cluster_rep_idx)

    extra_info = []
    for d in text_data:
        joint_info = len(gzip.compress(f"{text_data[rank1_idx]}\n{d}".encode()))
        extra_info.append(joint_info - gzip_info[rank1_idx])
    extra_info[rank1_idx] = -1
    rank2_idx = get_best_score_idx(extra_info, cluster_rep_idx)

    return rank1_idx, rank2_idx


def get_best_score_idx(scores: List[int], priority_idx: Union[int, None]) -> int:
    """Return the index of best score, prioritise priority_idx if a best scorer."""
    best_score = max(scores)
    if priority_idx is not None and scores[priority_idx] == best_score:
        return priority_idx
    else:
        return cast(int, np.argmax(scores))


def add_line_to_csv(row: Tuple[str, str, str, int], out_file_name: str) -> None:
    """Append a new row to csv file."""
    with open(out_file_name, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)
