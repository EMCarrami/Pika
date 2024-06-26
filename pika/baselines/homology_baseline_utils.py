import os
import pickle
import subprocess
from collections import defaultdict
from datetime import datetime
from typing import Dict, Literal

import pandas as pd
from loguru import logger

from pika.utils.helpers import file_path_assertions


def get_homologues_from_train_set(
    data_dict_path: str = "dataset/data_dict.pkl",
    split_path: str = "dataset/data_split.csv",
    split: Literal["val", "test"] = "val",
    blast_db: str = "dataset/blast_db/Pika_blast_db",
    out_path: str | None = None,
    save_blast_output_path: str | None = None,
) -> None:
    """Find the closest sequences to split sequences in the train set."""
    if out_path is None:
        out_path = f"dataset/split_homology_{split}.pkl"
    file_path_assertions(out_path, exists_ok=False)
    assert split in ["val", "test"], "only val/test splits are accepted."
    assert "." not in os.path.basename(blast_db) and not blast_db.endswith(
        "/"
    ), "blast_db must not be a file or folder. It should just name the blast db"

    if not os.path.isfile(f"{blast_db}.pdb"):
        logger.info(f"creating blast_db to {blast_db}")
        create_blast_db(data_dict_path, blast_db)

    with open(data_dict_path, "rb") as f:
        data_dict = pickle.load(f)
    seq_dict = {k: v["sequence"] for k, v in data_dict.items()}
    split_df = pd.read_csv(split_path)
    train_set = set(split_df[split_df.split == "train"]["uniprot_id"].to_list())
    assert len(train_set) > 0, f"no train values were found in {split_df}"
    assert split in split_df.split.unique(), f"no {split} values were found in {split_df}"
    split_df = split_df[split_df.split == split]

    # create query fasta
    time_stamp = datetime.now().strftime("%y%m%d%H%M%S")
    query_file_name = f"temp_query_{split}_{time_stamp}.fa"
    write_fasta({k: seq_dict[k] for k in split_df["uniprot_id"].to_list()}, query_file_name)
    # perform blastp
    save_path = f"temp_blastp_{split}_{time_stamp}.out" if save_blast_output_path is None else save_blast_output_path
    command = f'blastp -query {query_file_name} -db {blast_db} -out {save_path} -outfmt "6 qseqid sseqid score" '
    subprocess.run(command, shell=True, check=True)

    out = pd.read_csv(save_path, sep="\t", header=None)
    hmg, scores = defaultdict(list), defaultdict(list)
    for _, r in out.iterrows():
        if r[1] in train_set:
            hmg[r[0]].append(r[1])
            scores[r[0]].append(r[2])
    # sort by scores
    for k in hmg:
        hmg[k] = [x for x, _ in sorted(zip(hmg[k], scores[k]), key=lambda pair: pair[1], reverse=True)]

    with open(out_path, "wb") as f:
        pickle.dump(hmg, f)
    logger.info(f"train homologues of {split} set were saved in {out_path}")

    if save_blast_output_path is None:
        os.remove(save_path)
    else:
        logger.info(f"blastp results for {split} set were stored in {save_path}")


def create_blast_db(
    data_dict_path: str = "dataset/data_dict.pkl", out_path: str = "dataset/blast_db/Pika_blast_db"
) -> None:
    """Create a local BLAST database based on data_dict."""
    assert "." not in out_path and not out_path.endswith(
        "/"
    ), "out_path must not be a file or folder. It should just name the blast db"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(data_dict_path, "rb") as f:
        data_dict = pickle.load(f)
    seq_dict = {k: v["sequence"] for k, v in data_dict.items()}
    time_stamp = datetime.now().strftime("%y%m%d%H%M%S")
    write_fasta(seq_dict, f"{time_stamp}_temp.fa")
    command = f"makeblastdb -in {time_stamp}_temp.fa -dbtype prot -out {out_path}"
    subprocess.run(command, shell=True, check=True)
    os.remove(f"{time_stamp}_temp.fa")


def write_fasta(input_dict: Dict[str, str], file_name: str) -> None:
    """Create fasta files from input dictionary mapping IDs to sequences."""
    with open(file_name, "w") as fasta_file:
        for protein_id, sequence in input_dict.items():
            fasta_file.write(f">{protein_id}\n")
            fasta_file.write(f"{sequence}\n")


if __name__ == "__main__":
    get_homologues_from_train_set(
        data_dict_path="../../dataset/data_dict.pkl",
        split_path="evo_split_seed_1.csv",
        split="val",
        blast_db="../../dataset/blast_db/Pika_blast_db",
        out_path="evo_split_homology_val.pkl",
        save_blast_output_path=None,
    )
