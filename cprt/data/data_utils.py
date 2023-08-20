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

KEEP_FIELDS = [
    "GO",
    "Pfam",
    "InterPro",
    "Gene3D",
    "SUPFAM",
    "catalytic activity",
    "cofactor",
    "subunit",
    "subcellular location",
    "biophysicochemical properties",
]


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
    return len([k for k in d if k in KEEP_FIELDS])


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

    out_file_name = f"{DATA_PATH}/gzip_subsample_uniref{uniref_identity}.csv"
    with open(out_file_name, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("cluster_id", "uniprot_id", "info_fields", "rank_in_cluster"))

    for uniref, uniprots in tqdm(uniref_dict.items()):
        # get data for valid uniprot_ids in cluster
        (
            id_list,
            all_data,
            info_fields,
            gzip_info,
            cluster_rep,
        ) = get_uniref_cluster_data(uniprots, uniref)

        if len(id_list) > 1:
            best_score = max(gzip_info)
            if cluster_rep is not None and gzip_info[cluster_rep] == best_score:
                rank1_idx = cluster_rep
            else:
                rank1_idx = cast(int, np.argmax(gzip_info))

            extra_info = []
            for d in all_data:
                joint_info = len(gzip.compress(f"{all_data[rank1_idx]}\n{d}".encode()))
                extra_info.append(joint_info - gzip_info[rank1_idx])

            best_score = max(extra_info)
            if (
                cluster_rep is not None
                and cluster_rep != rank1_idx
                and extra_info[cluster_rep] == best_score
            ):
                rank2_idx = cluster_rep
            else:
                rank2_idx = cast(int, np.argmax(extra_info))

            add_line_to_csv(
                (uniref, id_list[rank1_idx], info_fields[rank1_idx], 1), out_file_name
            )
            add_line_to_csv(
                (uniref, id_list[rank2_idx], info_fields[rank2_idx], 2), out_file_name
            )
        else:
            add_line_to_csv((uniref, id_list[0], info_fields[0], 0), out_file_name)


def get_uniref_cluster_data(
    id_list: List[str], uniref_id: str
) -> Tuple[
    Tuple[str, ...], Tuple[str, ...], Tuple[str, ...], Tuple[int, ...], Union[int, None]
]:
    """Collect and filter the data for each uniprot_id."""
    all_data, info_fields, gzip_info, lengths = [], [], [], []
    cluster_rep_name = ""

    for idx, n in enumerate(id_list):
        cluster_rep_name = n if n in uniref_id else cluster_rep_name
        with open(f"{DATA_PATH}/swiss_prot/{n}.pkl", "rb") as f:
            info = pickle.load(f)
        fields = [k for k in info if k in KEEP_FIELDS]
        info_fields.append("; ".join(fields))
        keep_data = [f'{k}: {";".join(info[k])}' for k in fields]
        all_data.append("\n".join(keep_data))
        gzip_info.append(len(gzip.compress(all_data[-1].encode())))
        lengths.append(info["length"])

    median_len: float = np.median(lengths)
    len_ratios = np.array(lengths) / median_len
    id_list, all_data, info_fields, gzip_info = zip(
        *[
            (n, d, f, g)
            for n, d, f, g, l in zip(
                id_list, all_data, info_fields, gzip_info, len_ratios
            )
            if l > 0.25
        ]
    )

    cluster_rep = (
        id_list.index(cluster_rep_name) if cluster_rep_name in id_list else None
    )

    return id_list, all_data, info_fields, gzip_info, cluster_rep


def add_line_to_csv(row: Tuple[str, str, str, int], out_file_name: str) -> None:
    """Append a new row to csv file."""
    with open(out_file_name, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)
