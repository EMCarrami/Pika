import gzip
import pickle
from copy import deepcopy
from typing import Dict, List, Tuple, Union, cast

import numpy as np
import pandas as pd
from tqdm import tqdm

from cprt.data.data_utils import file_path_assertions

BASE_FIELDS = ["taxonomy", "protein size"]
INFO_FIELDS = [
    "functional domains",
    "catalytic activity",
    "cofactor",
    "subunit",
    "subcellular location",
    "pH dependence",
    "temperature dependence",
]


def subsample_uniref50_by_gzip_score(
    path_to_uniprot_data_dict: str = "dataset/swissprot_data_2023_08_14.pkl",
    path_to_uniref_mapping: str = "dataset/uniref50_dict.pkl",
    out_file_path: str = "dataset/uniref50_gzip_subsample_data.pkl",
) -> None:
    """
    Subsample uniref custer to maximum of two high-information representatives.

    :param path_to_uniprot_data_dict: dict mapping uniprot_ids to respective data fields.
    :param path_to_uniref_mapping: dict mapping each uniref50 cluster to a list of uniprot_ids
    :param out_file_path: path to output file. Must not already exist.
    :return: None. Saves the pickled dict for filtered entries with additional field for uniref_id
    """
    file_path_assertions(out_file_path, exists_ok=False)
    with open(path_to_uniref_mapping, "rb") as f:
        uniref_dict = pickle.load(f)
    merged_uniref_dict = merge_clusters(uniref_dict)

    with open(path_to_uniprot_data_dict, "rb") as f:
        all_uniprot_data = pickle.load(f)

    dataset = {}
    for uniref, uniprots in tqdm(merged_uniref_dict.items()):
        df, cluster_rep_uid = get_uniref_cluster_data(uniprots, uniref, all_uniprot_data)

        id_list = df["uniprot_id"].to_list()
        data_dicts = df["data_dict"].to_list()
        text_data = df["text_data"].to_list()
        gzip_info = df["gzip_info"].to_list()

        if len(id_list) > 1:
            top_ranks = get_top_info_ids(id_list, text_data, gzip_info, cluster_rep_uid)
            for r, idx in enumerate(top_ranks):
                uid = id_list[idx]
                dataset[uid] = data_dicts[idx]
        else:
            dataset[id_list[0]] = data_dicts[0]

    with open(out_file_path, "wb") as f:
        pickle.dump(dataset, f)


def merge_clusters(uniref_dict: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    Merge uniref clusters to reduce isoform-clusters.

    When isoforms are the representative of a cluster, will merge this isoform-cluster with the main protein's cluster.
    This makes the clustering more stringent by ensuring we won't take too many samples because of isoforms

    :param uniref_dict: dict mapping uniref_ids to a list of uniprot_ids
    :return: dict mapping merged uniref_ids to the merged list of uniprot_ids
    """
    new_dict = deepcopy(uniref_dict)
    # move all proteins clustered in an isoform-named-cluster to the main cluster of that isoform
    to_del = []
    for k, v in new_dict.items():
        if "-" in k:
            # k = Unirefxx_yyyyy-n
            base_cluster = k.split("-")[0]
            if base_cluster in new_dict:
                # when main protein is the representative of another cluster: merge in the isoform cluster
                new_dict[base_cluster] += v
                to_del.append(k)
            else:
                # when main protein is not the representative of any clusters: Check if it is a member of other clusters
                for _k, _v in new_dict.items():
                    # base_cluster = Unirefxx_yyyyy
                    if base_cluster.split("_")[1] in _v and base_cluster not in _k and _k not in to_del:
                        new_dict[_k] += v
                        to_del.append(k)
                        break
    for k in to_del:
        del new_dict[k]

    # ensure process was error-free
    new_size, original_size = 0, 0
    for v in uniref_dict.values():
        original_size += len(v)
    for v in new_dict.values():
        new_size += len(v)
    assert new_size == original_size

    return new_dict


def get_uniref_cluster_data(
    id_list: List[str], uniref_id: str, uniprot_data: Dict[str, Dict[str, List[str]]]
) -> Tuple[pd.DataFrame, str]:
    """
    Gather all the data and gzip_info value for each uniprot_id in the uniref cluster.

    Will exclude sequences that are shorter than median_len / 4 of the cluster.

    :param id_list: list of all uniprot_ids in the uniref cluster
    :param uniref_id: uniref cluster id
    :param uniprot_data: Dict of all uniprot data

    :returns: Two outputs
            - filtered DataFrame with columns uniprot_id, data_dict, text_data, gzip_info
            - uid of cluster_representative if among uniprot_ids, otherwise ''
    """
    # data_rows: uniprot_id, data_dict, text_data, gzip_info
    data_rows = []
    lengths = []
    cluster_rep_uid = ""

    for uid in id_list:
        cluster_rep_uid = uid if uid in uniref_id else cluster_rep_uid
        info_dict = uniprot_data[uid]
        assert isinstance(info_dict["sequence"], list) and len(info_dict["sequence"]) == 1
        lengths.append(len(info_dict["sequence"][0]))
        # text of info_fields for gzip info
        info_fields = [k for k in info_dict if k in INFO_FIELDS]
        text_data = "\n".join([f'{k}: {"; ".join(set(info_dict[k]))}' for k in info_fields])
        gzip_info = len(gzip.compress(text_data.encode()))
        # keep all fields for final data_dict (set to remove duplications)
        data_dict: Dict[str, List[str] | str] = {k: list(set(info_dict[k])) for k in info_fields}
        data_dict |= {k: info_dict[k] for k in BASE_FIELDS}
        data_dict["sequence"] = info_dict["sequence"][0]
        data_dict["uniref_id"] = uniref_id

        data_rows.append(
            (
                uid,
                data_dict,
                text_data,
                gzip_info,
            )
        )

    median_len = np.median(lengths)
    len_ratios = np.array(lengths) / median_len

    df = pd.DataFrame(
        data_rows,
        columns=["uniprot_id", "data_dict", "text_data", "gzip_info"],
    )
    df = df[len_ratios > 0.25]
    df.reset_index(drop=True, inplace=True)

    return df, cluster_rep_uid


def get_top_info_ids(
    id_list: List[str], text_data: List[str], gzip_info: List[int], cluster_rep_uid: str
) -> Tuple[int, int]:
    """
    Find the top ranking indices of data with highest gzip info.

    First picks the uid with the highest gzip information.
    Then computes the additional gzip information that each of the other uids provide in respect to the top one.
    Finally, selects the uid with the highest additional gzip information.
    In all steps, prioritises the cluster representative when there is a tie.
    """
    cluster_rep_idx = id_list.index(cluster_rep_uid) if cluster_rep_uid in id_list else None

    rank1_idx = get_best_score_idx(gzip_info, cluster_rep_idx)

    extra_info = []
    for d in text_data:
        joint_info = len(gzip.compress(f"{text_data[rank1_idx]}\n{d}".encode()))
        extra_info.append(joint_info - gzip_info[rank1_idx])
    extra_info[rank1_idx] = -1
    rank2_idx = get_best_score_idx(extra_info, cluster_rep_idx)

    return rank1_idx, rank2_idx


def get_best_score_idx(scores: List[int], priority_idx: Union[int, None]) -> int:
    """Return the index of best score, prioritise priority_idx when there is a tie."""
    best_score = max(scores)
    if priority_idx is not None and scores[priority_idx] == best_score:
        return priority_idx
    else:
        return cast(int, np.argmax(scores))
