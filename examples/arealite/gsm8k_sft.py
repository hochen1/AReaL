import os
import sys

import torch
from datasets import Dataset, load_dataset
from datasets.distributed import split_dataset_by_node
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import DataCollatorWithPadding

from arealite.api.cli_args import SFTConfig, load_expr_config
from arealite.api.io_struct import FinetuneSpec
from arealite.engine.fsdp_engine import FSDPEngine
from arealite.trainer.sft import SFTTrainer
from arealite.utils.data import pad_sequences_to_tensors
from realhf.api.core.data_api import load_hf_tokenizer


def process_gsm8k_sft_dataset(dataset: Dataset, tokenizer):
    def process(sample):
        seq_token = tokenizer.encode(
            sample["question"] + sample["answer"] + tokenizer.eos_token
        )
        prompt_token = tokenizer.encode(sample["question"])
        prompt_mask = [1] * len(prompt_token) + [0] * (
            len(seq_token) - len(prompt_token)
        )
        return {"input_ids": seq_token, "prompt_mask": prompt_mask}

    dataset = dataset.map(process).remove_columns(["question", "answer"])
    return dataset


def get_gsm8k_dataset(split, tokenizer, rank, world_size):
    dataset = load_dataset(path="openai/gsm8k", name="main", split=split)
    dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)
    return process_gsm8k_sft_dataset(dataset, tokenizer)


def main_sft():
    config, _ = load_expr_config(sys.argv[1:], SFTConfig)
    config: SFTConfig

    rank = int(os.getenv("RANK"))
    world_size = int(os.getenv("WORLD_SIZE"))
    tokenizer = load_hf_tokenizer(config.trainer.tokenizer_path)

    # Create dataset and dataloaders
    assert config.train_dataset == "gsm8k-sft"
    train_dataloader = StatefulDataLoader(
        get_gsm8k_dataset("train", tokenizer, rank, world_size),
        batch_size=config.train_dataset.batch_size // world_size,
        shuffle=config.train_dataset.shuffle,
        num_workers=config.train_dataset.num_workers,
        collate_fn=pad_sequences_to_tensors,
        drop_last=config.train_dataset.drop_last,
    )
    assert config.valid_dataset == "gsm8k-sft"
    valid_dataloader = StatefulDataLoader(
        get_gsm8k_dataset("test", tokenizer, rank, world_size),
        batch_size=config.valid_dataset.batch_size // world_size,
        shuffle=config.valid_dataset.shuffle,
        num_workers=config.valid_dataset.num_workers,
        collate_fn=pad_sequences_to_tensors,
        drop_last=config.valid_dataset.drop_last,
    )

    # Initialize engine
    ft_spec = FinetuneSpec(
        total_train_epochs=config.trainer.exp_ctrl.total_train_epochs,
        dataset_size=len(train_dataloader),
        train_batch_size=config.train_dataset.batch_size,
    )
    engine = FSDPEngine(config=config.model)
    engine.initialize(None, ft_spec)

    # Run training.
    trainer = SFTTrainer(
        config=config.trainer,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        engine=engine,
        inf_engine=None,
    )
    trainer.train()


if __name__ == "__main__":
    main_sft()
