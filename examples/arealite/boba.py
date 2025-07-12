import asyncio
import os
import re
import sys
import uuid

import torch
import torch.distributed as dist
from datasets import Dataset, load_dataset
from datasets.distributed import split_dataset_by_node
from tensordict import TensorDict
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizerFast

from arealite.api.cli_args import (
    GenerationHyperparameters,
    GRPOConfig,
    load_expr_config,
)
from arealite.api.io_struct import (
    FinetuneSpec,
    LLMRequest,
    LLMResponse,
    WeightUpdateMeta,
)
from arealite.api.workflow_api import RolloutWorkflow
from arealite.engine.ppo.actor import FSDPPPOActor
from arealite.engine.sglang_remote import RemoteSGLangEngine
from arealite.utils.data import concat_padded_tensors
from arealite.utils.evaluator import Evaluator
from arealite.utils.saver import Saver
from arealite.utils.stats_logger import StatsLogger
from realhf.api.core.data_api import load_hf_tokenizer
from realhf.base import stats_tracker


class RLVRWorkflow(RolloutWorkflow):
    def __init__(
        self,
        reward_fn,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast,
    ):
        self.reward_fn = reward_fn
        self.gconfig = gconfig
        self.tokenizer = tokenizer

    async def arun_episode(self, engine, data):
        input_ids = self.tokenizer.encode(data["prompt"])
        n_samples = self.gconfig.n_samples
        req = LLMRequest(
            rid=uuid.uuid4().hex,
            input_ids=input_ids,
            gconfig=self.gconfig.new(n_samples=1),
        )
        resps = await asyncio.gather(*[engine.agenerate(req) for _ in range(n_samples)])

        results = []
        for resp in resps:
            seq = resp.input_tokens + resp.output_tokens
            logprobs = [0] * resp.input_len + resp.output_logprobs
            prompt_mask = [1] * resp.input_len + [0] * resp.output_len
            versions = [-1] * resp.input_len + resp.output_versions

            reward = self.reward_fn(
                completions=self.tokenizer.decode(resp.output_tokens),
                prompt_ids=resp.input_tokens,
                completion_ids=resp.output_tokens,
                **data,
            )
            res = dict(
                # unsqueeze to add an additional batch dimension
                input_ids=torch.tensor(seq).unsqueeze(0),
                prompt_mask=torch.tensor(prompt_mask).unsqueeze(0),
                logprobs=torch.tensor(logprobs).unsqueeze(0),
                versions=torch.tensor(versions).unsqueeze(0),
                attention_mask=torch.ones(len(seq)).unsqueeze(0),
                # reward
                rewards=torch.tensor([reward]),
            )
            results.append(TensorDict(res, batch_size=[1]))

        return concat_padded_tensors(results)


def get_boba_math_dataset(rank, world_size):
    dataset = load_dataset(
        path="json",
        split="train",
        data_files="/storage/openpsi/users/xushusheng.xss/training_data/boba_106k_0319.jsonl",
    )
    return split_dataset_by_node(dataset, rank=rank, world_size=world_size)


def boba_reward_fn(
    prompt, completions, prompt_ids, completion_ids, query_id, solutions, **kwargs
):
    from realhf.impl.dataset.math_parser import process_results

    label = 0
    for sol in solutions:
        label = label or process_results(completions, sol)[0]
    return label


def main_grpo():
    config, _ = load_expr_config(sys.argv[1:], GRPOConfig)
    config: GRPOConfig

    rank = int(os.getenv("RANK"))
    world_size = int(os.getenv("WORLD_SIZE"))
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    # Create dataset and dataloaders
    train_dataloader = StatefulDataLoader(
        get_boba_math_dataset(rank, world_size),
        batch_size=config.train_dataset.batch_size // world_size,
        shuffle=config.train_dataset.shuffle,
        num_workers=config.train_dataset.num_workers,
        collate_fn=lambda x: x,
        drop_last=config.train_dataset.drop_last,
    )
    ft_spec = FinetuneSpec(
        total_train_epochs=config.total_train_epochs,
        dataset_size=len(train_dataloader) * config.train_dataset.batch_size,
        train_batch_size=config.train_dataset.batch_size,
    )

    # Initialize inference engine
    rollout = RemoteSGLangEngine(config.rollout)
    rollout.initialize(None, ft_spec)

    # Initialize train engine
    actor = FSDPPPOActor(config=config.actor)
    actor.initialize(None, ft_spec)
    ref = None
    if config.actor.kl_ctl > 0 and config.ref is not None:
        ref = FSDPPPOActor(config=config.ref)
        ref.initialize(None, ft_spec)

    # Create rollout workflow
    if tokenizer.pad_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.pad_token_id)
    if tokenizer.eos_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.eos_token_id)
    workflow = RLVRWorkflow(
        reward_fn=boba_reward_fn, gconfig=config.gconfig, tokenizer=tokenizer
    )

    # Run training.
    saver = Saver(config.saver, ft_spec, for_recover=False)
    logger = StatsLogger(config.stats_logger, ft_spec)
    evaluator = Evaluator(config.evaluator, ft_spec)

    total_epochs = config.total_train_epochs
    steps_per_epoch = len(train_dataloader)
    max_steps = total_epochs * steps_per_epoch

    logger.info(f"total_epochs={total_epochs} step_per_epoch={steps_per_epoch}")
    data_generator = iter(train_dataloader)
    for global_step in range(max_steps):
        epoch = global_step // steps_per_epoch
        step = global_step % steps_per_epoch

        with stats_tracker.record_timing("rollout"):
            if config.async_training:
                batch = rollout.prepare_batch(
                    data_generator,
                    train_dataloader,
                    workflow=workflow,
                )
            else:
                try:
                    data = next(data_generator)
                except StopIteration:
                    data_generator = iter(train_dataloader)
                    data = next(data_generator)
                batch = rollout.rollout(data, workflow=workflow)

        batch = batch.to(actor.device)

        if config.actor.recompute_logprob:
            with stats_tracker.record_timing("recompute_logp"):
                logp = actor.compute_logp(batch)
                if not config.actor.use_decoupled_loss:
                    batch["logprobs"] = logp
                else:
                    batch["prox_logp"] = logp

        if ref is not None:
            with stats_tracker.record_timing("ref_logp"):
                batch["ref_logp"] = ref.compute_logp(batch)

        with stats_tracker.record_timing("compute_advantage"):
            actor.compute_advantages(batch)

        with (
            stats_tracker.record_timing("train_step"),
            stats_tracker.scope("grpo_actor"),
        ):
            stats = actor.ppo_update(batch)
            actor.step_lr_scheduler()

        with stats_tracker.record_timing("update_weights"):
            meta = WeightUpdateMeta(
                type="disk",
                path=os.path.join(
                    Saver.get_save_checkpoint_root(config.saver), "update_weights"
                ),
                alloc_mode=None,
                comm_backend=None,
                model_version=global_step + 1,
            )
            if dist.get_rank() == 0:
                future = rollout.update_weights(meta)
            actor.upload_weights(meta)
            if dist.get_rank() == 0:
                future.result()
            rollout.set_version(global_step + 1)
            dist.barrier()

        with stats_tracker.record_timing("save"):
            saver.save(actor, epoch, step, global_step)

        logger.commit(epoch, step, global_step, stats)

    actor.destroy()
    if ref is not None:
        ref.destroy()
    rollout.destroy()
    logger.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main_grpo()
