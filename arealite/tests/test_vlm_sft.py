"""Test script for FSDP Engine implementation."""

from typing import Dict

import torch
from datasets import load_dataset

from arealite.api.cli_args import (
    DatasetConfig,
    EngineBackendConfig,
    EngineConfig,
    ModelFamily,
    OptimizerConfig,
    SFTTrainerConfig,
    TrainerConfig,
    TrainingArgs,
)
from arealite.api.trainer_api import TrainerFactory


def mock_loss_fn(logits: torch.Tensor, input_data: Dict) -> torch.Tensor:
    """Mock loss function for testing."""
    return torch.mean(logits)


def mock_loss_weight_fn(logits: torch.Tensor, input_data: Dict) -> float:
    """Mock loss weight function for testing."""
    return float(input_data["attention_mask"].sum())


def create_dataset(cfg: DatasetConfig):
    dataset = load_dataset(
        cfg.path,
        name=cfg.name,
        split=cfg.split,
    )
    return dataset


def test_engine():
    """Test engine creation and basic functionality."""

    train_dataset = DatasetConfig(
        path="/storage/openpsi/data/clevr_count_70k/",
        # name="main",
        split="train",
        batch_size=8,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
    )

    # valid_dataset = DatasetConfig(
    #     path="MathLLMs/MM-MathInstruct",
    #     # name="main",
    #     # split="test",
    #     batch_size=8,
    #     shuffle=False,
    #     pin_memory=True,
    #     num_workers=4,
    # )

    engine_config = EngineConfig(
        type=ModelFamily("qwen2_vl", False),
        path="/storage/openpsi/models/Qwen2-VL-7B",
        gradient_checkpointing=False,
        optimizer=OptimizerConfig(),
        backend=EngineBackendConfig(type="hf"),
    )

    sft_config = SFTTrainerConfig(
        model=engine_config,
    )

    train_config = TrainerConfig(
        type="sft",
        sft=sft_config,
    )

    args = TrainingArgs(
        experiment_name="vlm-test-sft",
        trial_name="test",
        mode="local",
        n_nodes=1,
        n_gpus_per_node=1,
        train_dataset=train_dataset,
        trainer=train_config,
    )

    rollout_controller = None
    train_dataset = create_dataset(args.train_dataset)
    valid_dataset = None
    if args.valid_dataset is not None:
        valid_dataset = create_dataset(args.valid_dataset)
    if args.trainer is not None:
        trainer_factory = TrainerFactory(args)
        trainer = trainer_factory.make_trainer(
            args.trainer,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            rollout_controller=rollout_controller,
        )
        trainer.train()

    print("All tests passed!")
