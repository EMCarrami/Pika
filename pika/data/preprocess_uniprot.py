import os
import pickle
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Union, cast

import numpy as np
import pandas as pd
import requests
from loguru import logger
from tqdm import tqdm

from pika.utils.helpers import file_path_assertions


def get_all_uniprot_info_in_parallel(
    path_to_swissprot_records: str = "dataset/uniprotkb_reviewed_true_2023_08_14.tsv",
    out_file_path: str = "dataset/swissprot_data_2023_08_14.pkl",
    min_protein_length: int = 30,
    ignore_no_info_entries: bool = True,
    batch_size: int = 10_000,
    num_workers: int = 20,
    temp_folder: str = "dataset/chunks",
) -> None:
    """
    Download and process the info for all uniprot_ids in a .tsv file and merge the final outputs.

    :param path_to_swissprot_records: Path to the tsv file containing uniprot_ids with 'Entry' and 'Length' as headers.
                                      Length will be used to filter out proteins shorter than min_protein_length.
    :param out_file_path: Path where the final merged dataset file will be saved.
                          The file must have a .pkl extension and not already exist.
    :param min_protein_length: Minimum length of proteins to be included in the dataset.
                               Proteins shorter than this will be excluded.
    :param ignore_no_info_entries: Whether to ignore entries without any info aside from taxonomy, size and sequence
    :param batch_size: Number of records to process in parallel in each batch.
    :param num_workers: Number of worker threads to use for parallel processing.
                        A value of 0 means processing will be done serially.
    :param temp_folder: Path to the folder where temporary chunk files will be stored during processing.

    :return: None. The function saves the output file at out_file_path.
    Output Example:
    {'A0A009IHW8': {
            'taxonomy': ['Bacteria', 'Pseudomonadota', 'Gammaproteobacteria'],
            'protein size': ['269 aa', '30922 KDa'],
            'sequence': ['MSLEQKKGADIISKILQIQNSIGKTTSP ... '],
            'catalytic activity': ['H2O + NAD(+) = ADP-D-ribose + H(+) + nicotinamide', 'EC = 3.2.2.6'],
            'subunit': ['Homodimer', conformational changes occur upon 3AD binding.'],
            'functional domains': ['NAD(P)+ nucleosidase activity', 'NAD+ nucleosidase activity', ...]
        }
    }
    """
    base_name, _ = file_path_assertions(out_file_path, exists_ok=False, strict_extension=".pkl")
    os.makedirs(temp_folder, exist_ok=True)

    try:
        df = pd.read_csv(path_to_swissprot_records, sep="\t")[["Entry", "Length"]]
    except KeyError as e:
        raise Exception(
            f"{e} swissprot_records file must be tab-separated with a header including 'Entry' and 'Length' columns."
        )
    df = df[df["Length"] >= min_protein_length]

    n_chunks = len(df) // batch_size + (1 if len(df) % batch_size else 0)
    df_chunks = np.array_split(df, n_chunks)
    uid_lists, all_temps, unprocessed_names = [], [], []
    for idx, chunk in enumerate(df_chunks):
        assert isinstance(chunk, pd.DataFrame)
        chunk_file_name = f"{temp_folder}/{base_name}_{idx}_{chunk.index.start}-{chunk.index.stop}.pkl"
        all_temps.append(chunk_file_name)
        if os.path.isfile(chunk_file_name):
            logger.info(f"chunk {idx} already processed in file {chunk_file_name} ... Skipping.")
        else:
            unprocessed_names.append(chunk_file_name)
            uid_lists.append(chunk["Entry"].to_list())

    if num_workers > 0:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            logger.info(f"submitting {len(uid_lists)} tasks to {num_workers} workers.")
            futures = [
                executor.submit(get_info_for_uniprot_ids, uids, fn) for uids, fn in zip(uid_lists, unprocessed_names)
            ]
            for future in tqdm(as_completed(futures), total=len(futures)):
                try:
                    future.result()
                except Exception as exc:
                    for f in futures:
                        f.cancel()
                    raise Exception from exc
    else:
        for uids, fn in zip(uid_lists, unprocessed_names):
            get_info_for_uniprot_ids(uids, fn)

    merged_dict = {}
    for fn in tqdm(all_temps):
        with open(fn, "rb") as file:
            data = pickle.load(file)
        if ignore_no_info_entries:
            # keep only entries that have more than the 3 fields of taxonomy, protein size, and sequence
            data = {k: v for k, v in data.items() if len(v) > 3}
        merged_dict.update(data)
    with open(out_file_path, "wb") as output_file:
        pickle.dump(merged_dict, output_file)


def get_info_for_uniprot_ids(
    uniprot_ids: List[str],
    out_file_path: str,
) -> None:
    """Download, pre-process and save data for a list of uniprot_ids from uniprot API."""
    file_path_assertions(out_file_path, exists_ok=True)

    batch = {}
    for idx in tqdm(uniprot_ids):
        batch[idx] = preprocess_uniprot(idx)
    with open(out_file_path, "wb") as f:
        pickle.dump(batch, f)


def preprocess_uniprot(uniprot_id: str) -> Union[Dict[str, List[str]], None]:
    """
    Get xml from uniprot and preprocess data fields.

    For each entry extracts the following:
    sequence (molecular mass and length)
    organism (top three taxonomic levels)
    catalytic activity (including EC number)
    biophysicochemical properties (pH and temperature dependence)
    cofactor
    subunit (excluding fields containing “interact”, “associate”, or “complex”)
    subcellular location (excluding isoforms)
    functional domains: GO (only molecular function, omitting biological process and cellular component),
                        Gene3D, and SUPFAM
    """
    out = defaultdict(list)
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.xml"
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Unexpected status code {response.status_code} for {uniprot_id}")
    ns = {"uniprot": "http://uniprot.org/uniprot"}
    root = ET.fromstring(response.content)
    entry = root.find(".//uniprot:entry", ns)
    if entry is None:
        return None

    # Find the 'organism' element and extract the first three 'taxon' value
    organism = entry.find("./uniprot:organism", ns)
    if organism is not None:
        taxa = organism.findall("./uniprot:lineage/uniprot:taxon", ns)
        out["taxonomy"] = [taxon.text for taxon in taxa[:3]]

    # Find the 'sequence' element and extract the 'length', 'mass', and sequence text
    sequence_elem = entry.find("./uniprot:sequence", ns)
    if sequence_elem is not None:
        out["protein size"] = [
            f"{out['length']} aa",
            f"{sequence_elem.get('mass')} KDa",
        ]
        out["sequence"] = [sequence_elem.text]
    else:
        return None

    ref_pattern = r"\(PubMed:\d+(?:, PubMed:\d+)*\)"
    subunit_ignore_kw = ["interact", "associate", "complex"]
    roi_ignore_kw = ["disordered", "interact", "required"]

    field_mapping: Dict[str, List[str]] = {
        "catalytic activity": [
            "./uniprot:reaction/uniprot:text",
            "./uniprot:reaction/uniprot:dbReference[@type='EC']",
        ],
        "biophysicochemical properties": [
            "pH dependence;./uniprot:phDependence/uniprot:text",
            "temperature dependence;./uniprot:temperatureDependence/uniprot:text",
        ],
        "cofactor": ["./uniprot:cofactor/uniprot:name", "./uniprot:text"],
        "subunit": ["./uniprot:text"],
        "subcellular location": [
            "./uniprot:subcellularLocation/uniprot:location",
            "./uniprot:subcellularLocation/uniprot:topology",
        ],
    }
    dbReference_types = ["GO", "Gene3D", "SUPFAM"]
    feature_types = [
        "short sequence motif",
        "domain",
        "binding site",
        "region of interest",
        "disulfide bond",
    ]

    # Find all 'comment' elements within each 'entry'
    for comment in entry.findall(".//uniprot:comment", ns):
        assert comment is not None
        comment_type = comment.get("type")
        if comment_type in field_mapping:
            # Ignore subunit values starting with "("
            if comment_type == "subunit":
                element = comment.find("./uniprot:text", ns)
                assert element is not None
                text = element.text
                assert isinstance(text, str)
                if text.startswith("("):
                    continue
                sntc = [
                    re.sub(ref_pattern, "", i).strip()
                    for i in text.split(". ")
                    if not any([j in i.lower() for j in subunit_ignore_kw])
                ]
                if len(sntc) > 0:
                    out[comment_type] += sntc
            # pH and temp dependence
            elif comment_type == "biophysicochemical properties":
                for k_f in field_mapping[comment_type]:
                    k, f = k_f.split(";")
                    field = comment.find(f, ns)
                    if field is not None:
                        out[k] = [field.text]
            # Ignore subcellular location from isoforms
            elif comment_type == "subcellular location" and comment.find("./uniprot:molecule", ns) is not None:
                continue
            else:
                for field_path in field_mapping[comment_type]:
                    # If the field path is for the EC number, extract the 'id' attribute
                    if "[@type='EC']" in field_path:
                        dbReference = comment.find(field_path, ns)
                        if dbReference is not None:
                            value = dbReference.get("id")
                            out[comment_type].append(f"EC = {value}")
                    else:
                        for field in comment.findall(field_path, ns):
                            out[comment_type].append(re.sub(ref_pattern, "", cast(str, field.text)))

    # Extract 'dbReference' elements
    for dbReference in entry.findall(".//uniprot:dbReference", ns):
        dbReference_type = dbReference.get("type")
        if dbReference_type in dbReference_types:
            if dbReference_type == "GO":
                element = dbReference.find("./uniprot:property[@type='term']", ns)
                if element is not None:
                    element_value = element.get("value")
                    if isinstance(element_value, str) and element_value.startswith("F:"):
                        out["functional domains"].append(element_value.replace("F:", ""))
            else:
                element = dbReference.find("./uniprot:property[@type='entry name']", ns)
                if element is not None:
                    out["functional domains"].append(element.get("value"))

    for feature in entry.findall(".//uniprot:feature", ns):
        feature_type = feature.get("type")
        if feature_type in feature_types:
            description = feature.get("description")
            # Ignore 'region of interest' with description 'Disordered' or starting with 'required'
            if description is None or (
                feature_type == "region of interest" and any([i in description.lower() for i in roi_ignore_kw])
            ):
                continue
            begin_elem = feature.find("./uniprot:location/uniprot:begin", ns)
            end_elem = feature.find("./uniprot:location/uniprot:end", ns)
            position_elem = feature.find("./uniprot:location/uniprot:position", ns)
            if begin_elem is not None and end_elem is not None:
                location = [begin_elem.get("position"), end_elem.get("position")]
            elif position_elem is not None:
                location = [position_elem.get("position")]
            else:
                location = []
            if feature_type == "binding site":
                element = feature.find("./uniprot:ligand/uniprot:name", ns)
                assert element is not None
                ligand_name = element.text
                out[feature_type].append(f"{ligand_name}: {'-'.join(location)}")  # type: ignore[arg-type]
            else:
                out[feature_type].append(f"{description}: {'-'.join(location)}")  # type: ignore[arg-type]
    return out  # type: ignore[return-value]
