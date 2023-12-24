# cprt
chat

## Standard Install

```bash
yes | conda create --name cprt python=3.10
conda activate cprt
pip install -e .[dev]
pre-commit install
```

## New env Install

To install conda
```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh
```

install requirements
```bash
cd PertPrompt
yes | conda create --name rpp python=3.10
conda activate rpp
pip install -r requirements/requirements_dev.txt
pip install -e .
yes | conda install ipykernel
python -m ipykernel install --user --name=rpp
export JUPYTER_PATH=/home/jovyan/.local/share/jupyter
```

Then restart the notebook
