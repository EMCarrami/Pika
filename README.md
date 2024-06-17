<h1 id="pika">Pika <img src="assets/images/Pika_logo.png" alt="Pika Framework" height="50" align="top"></h1>

![Code License](https://img.shields.io/badge/Code%20License-MIT-green.svg)
![Data License](https://img.shields.io/badge/Data%20License-CC%20BY%204.0-red.svg)



- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Checkpoints](#model-checkpoints)
- [Disclaimer](#disclaimer)

## Introduction

Pika enables scientific question answering about protein sequences.

In [PQA: Zero-shot Protein Question Answering for Free-form Scientific Enquiry with Large Language Models](https://arxiv.org/pdf/2402.13653) we introduce the novel task of zero-shot Protein Question Answering (PQA) for free-form scientific enquiry. Given a previously unseen protein sequence and a natural language question, the task is to deliver a scientifically accurate answer. 

<p align="left"><img src="assets/images/Pika.png" title="Pika Framework" height="500"></p>

## Installation

```bash
pip install git+https://github.com/EMCarrami/Pika.git
```

From source
```bash
yes | conda create --name pika python=3.10
conda activate pika
pip install -e .
```

For contribution please install in dev mode and use pre-commit
```bash
yes | conda create --name pika python=3.10
conda activate pika
pip install -e ".[dev]"
pre-commit install
```

## Usage

### Commandline

```bash
python run.py --config configs/train_config.json --run_mode train_and_benchmark
```

Nested keys can be parsed as below as long as all keys are present in the config
```bash
python run.py --config configs/train_config.json --run_mode train_and_benchmark --model.enable_gradient_checkpointing True
```

### Python

Please check [notebooks](https://github.com/EMCarrami/Pika/tree/main/notebooks) for examples.

For training followed by Biochem-ReAct benchmarking
```python
from pika.main import Pika
from pika.utils.helpers import load_config

config = load_config("path/to/config")
model = Pika(config)
model.train()
model.biochem_react_benchmark(model_to_use="best")
```

## Dataset

Complete Pika-DS is available on [HuggingFace](https://huggingface.co/datasets/EMCarrami/Pika-DS)

## Model Checkpoints 
See [example notebook](https://github.com/EMCarrami/Pika/blob/main/notebooks/scientific_enquiry.ipynb) for usage

| model_type | LLM   | PLM | split basis                  | checkpoint_file                    | partial* |
|------------|-------|-----|------------------------------|------------------------------------|---------|
| Self-Pika  | Phi-2 | ESM2-t33-650M | [UniRef50](https://huggingface.co/datasets/EMCarrami/Pika-DS/blob/main/splits/pika_uniref_split.csv) | [self-pika-phi2-esm2-t33-uniref.ckpt](https://github.com/EMCarrami/Pika/blob/main/model_checkpoints/self-pika-phi2-esm2-t33-uniref.ckpt) | Yes     |

\* "Partial" indicates whether the model checkpoint is only for trained weights (without frozen Phi-2 and ESM2 weights) or the entire model. When partail==Yes, Pika object automatically retrieves pre-trained weights for Phi-2 and ESM2 from HuggungFace.

## Disclaimer

All data and model checkpoints for Pika are licensed under CC BY 4.0, permitting non-commercial use. Pika-DS is based on UniProt database and any restricts that apply to UniProt also apply to Pika-DS. Pika model checkpoints are based on pre-trained Phi-2 and ESM2 models. All respective restrictions also apply to Pika models.

We developed Pika-DS, utilizing the publicly available SwissProt database and processing with GPT3.5. Given the limitations of Large Language Models (LLMs) in generating large-scale synthetic datasets, we endeavored to minimize the inclusion of harmful content in Pika-DS through prompt optimization and manual evaluation. Nonetheless, due to the dataset’s extensive size, there’s a slight possibility that unintended harmful content might still be present. Our Pika-based pretrained models are derived from the publicly accessible and unmoderated Phi-2. Thus, all cautions, restrictions, and notes associated with phi-2 are applicable to our models. The Pika framework is specifically designed for question answering related to protein sequences. With scientists having identified nearly 0.25 billion protein sequences, and functional annotations available for fewer than a million, our framework offers potentials for research into these largely unexplored proteins. While our efforts are directed towards scientific research, we recognize the potential risk of misuse by individuals aiming to create or identify harmful substances. **We strictly prohibit using our dataset and framework for any illegal activities or actions that could harm individuals or society.**
