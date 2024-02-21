# Pika

<table width="10%" border="0" cellspacing="0" cellpadding="0">
  <tr>
    <td style="font-size: 16px; width: 82%; padding-top: 0; padding-bottom: 0;">
      Pika enables scientific question answering about protein sequences.
      <br>
      <br>
      <a href="https://arxiv.org/abs/XXXXXX">PQA: Zero-shot Protein Question Answering for Free-form Scientific Enquiry with Large Language Models</a>
    </td>
    <td style="padding-top: 0; padding-bottom: 0;"><img src="assets/Pika_logo.png" title="Pika Framework" height="150" style="float: right;"></td>
  </tr>
</table>


<p align="left"><img src="assets/Pika.png" title="Pika Framework" height="500"></p>

## Installation from source

```bash
pip install pip install git+https://github.com/EMCarrami/Pika.git
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
