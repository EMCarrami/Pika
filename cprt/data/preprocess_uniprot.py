import os
import pickle
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Dict, List, Union, cast

import requests
from tqdm import tqdm

from cprt.utils import DATA_PATH


def preprocess_uniprot(uniprot_id: str) -> Union[Dict[str, List[str]], None]:
    """Get xml from uniprot and preprocess data fields."""
    out = defaultdict(list)
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.xml"
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(
            f"Unexpected status code {response.status_code} for {uniprot_id}"
        )
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
        out["length"] = [sequence_elem.get("length")]
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
            elif (
                comment_type == "subcellular location"
                and comment.find("./uniprot:molecule", ns) is not None
            ):
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
                            out[comment_type].append(
                                re.sub(ref_pattern, "", cast(str, field.text))
                            )

    # Extract 'dbReference' elements
    for dbReference in entry.findall(".//uniprot:dbReference", ns):
        dbReference_type = dbReference.get("type")
        if dbReference_type in dbReference_types:
            if dbReference_type == "GO":
                element = dbReference.find("./uniprot:property[@type='term']", ns)
                if element is not None:
                    element_value = element.get("value")
                    if isinstance(element_value, str) and element_value.startswith(
                        "F:"
                    ):
                        out["functional domains"].append(
                            element_value.replace("F:", "")
                        )
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
                feature_type == "region of interest"
                and any([i in description.lower() for i in roi_ignore_kw])
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


def get_uniprotkb_reviewed_info_in_chunks(
    chunk: int,
    chunk_size: int = 100_000,
    batch_size: int = 1000,
    file_date: str = "2023_08_14",
    out_folder_name: str = "data_chunks",
) -> None:
    """Store pre-processed data from uniprot API based on uniprotkb_reviewed_true_{date}.tsv list."""
    tracker_file_name = f"{DATA_PATH}/{out_folder_name}/chunk_{chunk}_processed_ids.tsv"
    chunks_path = f"{DATA_PATH}/{out_folder_name}/chunks"
    os.makedirs(chunks_path, exist_ok=True)

    total_rows = 570_000
    max_num_chunks = total_rows // chunk_size + 1
    assert 0 < chunk < max_num_chunks + 1, (
        f"chunk can be between 1 and {max_num_chunks}. "
        f"Each chunk covers {chunk_size} entries"
    )
    start_line = (chunk - 1) * 100_000 + 1
    end_line = chunk * 100_000

    ids, lens = [], []
    with open(f"{DATA_PATH}/uniprotkb_reviewed_true_{file_date}.tsv", "r") as f:
        for line_number, line in enumerate(f):
            if start_line <= line_number <= end_line:
                ids.append(line.split("\t")[0])
                lens.append(int(line.split("\t")[-1]))

    batch_count = 0
    processed_ids = []
    if os.path.isfile(tracker_file_name):
        with open(tracker_file_name, "r") as f:
            for line in f:
                processed_ids.append(line.split("\t")[0])
                batch_count = int(line.split("\t")[-1])
        print(
            f"found {len(processed_ids)} processed ids and {batch_count} batches for chunk {chunk}"
        )

    batch, status = {}, []
    batch_count += 1
    for idx, length in tqdm(zip(ids, lens), total=len(ids)):
        if idx in processed_ids:
            continue
        if length < 30:
            status.append([idx, "fail", 0, chunk, batch_count])
            continue

        values = preprocess_uniprot(idx)
        if values is None:
            status.append([idx, "fail", 0, chunk, batch_count])
        else:
            batch[idx] = values
            status.append([idx, "pass", len(values) - 4, chunk, batch_count])

        if len(batch) == batch_size:
            with open(
                f"{chunks_path}/chunk_{chunk}_batch_{batch_count}.pkl", "wb"
            ) as file:
                pickle.dump(batch, file)
            with open(tracker_file_name, "a") as f:
                for row in status:
                    f.write("\t".join([str(x) for x in row]))
                    f.write("\n")
            batch_count += 1
            batch, status = {}, []

    if len(batch) > 0:
        with open(f"{chunks_path}/chunk_{chunk}_batch_{batch_count}.pkl", "wb") as file:
            pickle.dump(batch, file)
        with open(tracker_file_name, "a") as f:
            for row in status:
                f.write("\t".join([str(x) for x in row]))
                f.write("\n")


def merge_uniprot_data_chunks(
    chunks_folder_path: str = f"{DATA_PATH}/data_chunks/chunks",
    out_file_path: str = f"{DATA_PATH}/merged_uniprot_data.pkl",
) -> None:
    """Merge all pickle files in chunks path to a new pickle file in out_path."""
    merged_dict = {}
    for file_name in tqdm(os.listdir(chunks_folder_path)):
        file_path = os.path.join(chunks_folder_path, file_name)
        if file_path.endswith(".pkl"):
            with open(file_path, "rb") as f:
                data = pickle.load(f)
            merged_dict.update(data)
    with open(out_file_path, "wb") as output_file:
        pickle.dump(merged_dict, output_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk", type=int, help="chunk id")

    args = parser.parse_args()
    chunk = args.chunk
    get_uniprotkb_reviewed_info_in_chunks(chunk)
