import random
from typing import Any, Dict, cast

import torch
from lightning.pytorch import seed_everything
from tqdm import tqdm
from transformers import AutoTokenizer, GPT2LMHeadModel

import wandb
from cprt.data.cprt_datamodule import CPrtDataModule
from cprt.metrics.biochem_metrics import BiochemMetrics
from cprt.model.original_phi.modeling_phi import PhiForCausalLM
from cprt.utils.helpers import load_config

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def get_random_baseline(config: Dict[str, Any]) -> None:
    """Compute metrics on randomly selected responses."""
    if "seed" in config:
        seed_everything(config["seed"])

    config["datamodule"]["language_model"] = config["model"]["language_model"]
    config["datamodule"]["protein_model"] = config["model"]["protein_model"]
    datamodule = CPrtDataModule(**config["datamodule"])

    metric = BiochemMetrics()
    df = datamodule.val_metric_dataset.split_df
    for n in df.metric.unique():
        lbl = df[df.metric == n]["value"].to_list()
        preds = [random.choice(["yes", "no"]) if isinstance(i, bool) else str(i) for i in lbl]
        random.shuffle(preds)
        metric.update(preds, lbl, [n] * len(lbl))

    config["wandb"]["group"] = "random_baseline"
    config["wandb"]["name"] = f"random_baseline_{config['seed']}"
    wandb.init(**config["wandb"], config=config)
    wandb.log({f"biochem/val_{k}": v for k, v in metric.compute().items()})
    wandb.finish()


def get_llm_only_baseline(config: Dict[str, Any]) -> None:
    """Compute metrics with only the LLM without input from proteins."""
    config["datamodule"]["sequence_placeholder"] = ""
    if "seed" in config:
        seed_everything(config["seed"])

    config["datamodule"]["language_model"] = config["model"]["language_model"]
    config["datamodule"]["protein_model"] = config["model"]["protein_model"]
    datamodule = CPrtDataModule(**config["datamodule"])

    if "gpt2" in config["model"]["language_model"]:
        text_tokenizer = AutoTokenizer.from_pretrained(config["model"]["language_model"], use_fast=False)
        llm = GPT2LMHeadModel.from_pretrained(config["model"]["language_model"]).to(device)
    elif "phi" in config["model"]["language_model"]:
        text_tokenizer = AutoTokenizer.from_pretrained(
            config["model"]["language_model"], use_fast=False, revision="7e10f3ea09c0ebd373aebc73bc6e6ca58204628d"
        )
        llm = PhiForCausalLM.from_pretrained(
            config["model"]["language_model"],
            flash_attn=True,
            flash_rotary=True,
            fused_dense=True,
            revision="7e10f3ea09c0ebd373aebc73bc6e6ca58204628d",
        ).to(device)
    else:
        raise ValueError("only gpt2 and microsoft/phi-2 models are supported.")

    metric = BiochemMetrics()
    generation_length = 20
    llm.eval()
    with torch.no_grad():
        for batch in tqdm(datamodule.val_dataloader()[1]):
            info_ids = batch.question.to(device)
            out = []
            if torch.any(info_ids == text_tokenizer.eos_token_id):
                for question in info_ids:
                    mask = question == text_tokenizer.eos_token_id
                    prompt_len = cast(int, torch.where(mask)[0][0] if mask.any() else len(question))
                    pos_offset = prompt_len
                    preds = llm.generate(
                        input_ids=question[:prompt_len].unsqueeze(0),
                        use_cache=True,
                        max_length=generation_length + prompt_len,
                    )
                    out.append(text_tokenizer.decode(preds[0, pos_offset:].cpu(), skip_special_tokens=True))
            else:
                prompt_len = info_ids.size(1)
                pos_offset = prompt_len
                preds = llm.generate(
                    input_ids=info_ids,
                    use_cache=False,
                    max_length=generation_length + prompt_len,
                )
                for pred in preds:
                    out.append(text_tokenizer.decode(pred[pos_offset:].cpu(), skip_special_tokens=True))
            metric.update(out, batch.expected_value, batch.metric_name)

    config["wandb"]["group"] = "llm_only_baseline"
    config["wandb"]["name"] = f"llm_only_baseline_{config['seed']}"
    wandb.init(**config["wandb"], config=config)
    wandb.log({f"biochem/val_{k}": v for k, v in metric.compute().items()})
    wandb.finish()


if __name__ == "__main__":
    c = load_config("baseline_config.json")
    if c["model"]["multimodal_strategy"] == "llm_only_baseline":
        get_llm_only_baseline(c)
    elif c["model"]["multimodal_strategy"] == "random_baseline":
        get_random_baseline(c)
    else:
        raise ValueError(
            f"{c['model']['multimodal_strategy']} not supported. "
            "Set mode.multimodal_strategy to llm_only_baseline or random_baseline"
        )
