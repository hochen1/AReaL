import getpass
import os
from typing import Dict

import torch.distributed as dist
import wandb
from tensorboardX import SummaryWriter
from torchdata.stateful_dataloader import StatefulDataLoader

from arealite.api.cli_args import TrainerConfig
from arealite.api.engine_api import InferenceEngine, TrainEngine
from realhf.api.core.data_api import load_hf_tokenizer
from realhf.base import logging, timeutil


class Trainer:
    def __init__(
        self,
        config: TrainerConfig,
        train_dataloader: StatefulDataLoader,
        valid_dataloader: StatefulDataLoader,
        engine: TrainEngine,
        inf_engine: InferenceEngine | None = None,
    ):
        self.config = config

        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader

        self.engine = engine
        self.inf_engine = inf_engine

        self.tokenizer = load_hf_tokenizer(config.tokenizer_path)

        self.save_ctl = timeutil.EpochStepTimeFreqCtl(
            freq_epoch=config.exp_ctrl.save_freq_epochs,
            freq_step=config.exp_ctrl.save_freq_steps,
            freq_sec=config.exp_ctrl.save_freq_secs,
        )
        self.eval_ctl = timeutil.EpochStepTimeFreqCtl(
            freq_epoch=config.exp_ctrl.eval_freq_epochs,
            freq_step=config.exp_ctrl.eval_freq_steps,
            freq_sec=config.exp_ctrl.eval_freq_steps,
        )
        self.logger = logging.getLogger(self.__class__.__name__)
        self.init_stats_logging()

    def init_stats_logging(self):
        """
        Initialize wandb and/or tensorboard according to config.
        If torch.distributed is initialized

        Return:
            tensorboard SummaryWriter if self.config.tensorboard.path is not None
        """
        if dist.is_initialized() and dist.get_rank() != 0:
            return

        # wandb init, connect to remote wandb host
        if self.config.wandb.mode != "disabled":
            wandb.login()
        wandb.init(
            mode=self.config.wandb.mode,
            entity=self.config.wandb.entity,
            project=self.config.wandb.project or self.config.experiment_name,
            name=self.config.wandb.name or self.config.trial_name,
            job_type=self.config.wandb.job_type,
            group=self.config.wandb.group
            or f"{self.config.experiment_name}_{self.config.trial_name}",
            notes=self.config.wandb.notes,
            tags=self.config.wandb.tags,
            config=self.config.wandb.config,
            dir=Trainer.get_log_path(self.config),
            force=True,
            id=f"{self.config.experiment_name}_{self.config.trial_name}_train",
            resume="allow",
            settings=wandb.Settings(start_method="fork"),
        )
        # tensorboard logging
        self.summary_writer = None
        if self.config.tensorboard.path is not None:
            self.summary_writer = SummaryWriter(log_dir=self.config.tensorboard.path)

    def log_wandb_tensorboard(self, step: int, data: Dict):
        if dist.is_initialized() and dist.get_rank() != 0:
            return

        wandb.log(data, step=step)
        if self.summary_writer is not None:
            for key, val in data.items():
                self.summary_writer.add_scalar(f"{key}", val, step)

    def close_wandb_tensorboard(self):
        if dist.is_initialized() and dist.get_rank() != 0:
            return

        wandb.finish()
        if self.summary_writer is not None:
            self.summary_writer.close()

    @staticmethod
    def get_save_checkpoint_path(
        config: TrainerConfig,
        epoch: int,
        step: int,
        globalstep: int,
        name: str = "default",
    ):
        path = os.path.join(
            f"{config.fileroot}/checkpoints/{getpass.getuser()}/{config.experiment_name}/{config.trial_name}",
            name,
            f"epoch{epoch}epochstep{step}globalstep{globalstep}",
        )
        os.makedirs(path, exist_ok=True)
        return path

    @staticmethod
    def get_log_path(config: TrainerConfig):
        path = f"{config.fileroot}/logs/{getpass.getuser()}/{config.experiment_name}/{config.trial_name}"
        os.makedirs(path, exist_ok=True)
        return path

    def log(self, msg: str, level="info"):
        if dist.is_initialized() and dist.get_rank() > 0:
            return
        log_fn = getattr(self.logger, level, "info")
        return log_fn(msg)

    def train(self):
        raise NotImplementedError()
