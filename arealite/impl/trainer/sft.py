import time

import torch
import torch.distributed as dist
from datasets import Dataset
from typing import Any, Dict, List, Optional, Tuple, Union
from arealite.api.cli_args import TrainerConfig, TrainingArgs
from arealite.api.engine_api import EngineFactory
from arealite.api.trainer_api import Trainer
from arealite.system.rollout_controller import RolloutController
from arealite.utils import (
    close_wandb_tensorboard,
    compute_varlen_position_indices,
    gather_logprobs,
    get_save_checkpoint_path,
    init_stats_logging,
    log_wandb_tensorboard,
    record_timing,
)
from realhf.api.core.data_api import load_hf_tokenizer, tabulate_stats, load_hf_processor_and_tokenizer
from realhf.api.core.model_api import FinetuneSpec
from realhf.base import logging, stats_tracker, timeutil
from arealite.impl.engine.constant import VALID_VISION_MODELS 
from PIL import Image
from PIL.Image import Image as ImageObject
from io import BytesIO

import math
logger = logging.getLogger("SFT Trainer")

# def process_image(
#     image: Union[Dict[str, Any], ImageObject, str], min_pixels: Optional[int]=None, max_pixels: Optional[int]=None
# ) -> ImageObject:
    
#     if isinstance(image, str):
#         image = Image.open(image)
#     elif isinstance(image, dict):
#         image = Image.open(BytesIO(image["bytes"]))
#     elif isinstance(image, bytes):
#         image = Image.open(BytesIO(image))

#     image.load()  # avoid "Too many open files" errors
#     if max_pixels is not None and (image.width * image.height) > max_pixels:
#         resize_factor = math.sqrt(max_pixels / (image.width * image.height))
#         width, height = int(image.width * resize_factor), int(image.height * resize_factor)
#         image = image.resize((width, height))

#     if min_pixels is not None and (image.width * image.height) < min_pixels:
#         resize_factor = math.sqrt(min_pixels / (image.width * image.height))
#         width, height = int(image.width * resize_factor), int(image.height * resize_factor)
#         image = image.resize((width, height))

#     if image.mode != "RGB":
#         image = image.convert("RGB")
#     return image

def compute_packed_sft_loss(
    logits: torch.Tensor,
    input_: Dict[str, torch.Tensor],
) -> torch.Tensor:
    packed_input_ids: torch.Tensor = input_["input_ids"].squeeze(dim=0)
    cu_seqlens: torch.Tensor = input_["cu_seqlens"]
    input_lens: torch.Tensor = cu_seqlens[1:] - cu_seqlens[:-1]
    cu_seqlens = torch.nn.functional.pad(input_lens.cumsum(0), (1, 0)).int()
    prompt_mask = input_["prompt_mask"].squeeze(dim=0)
    logits = logits.squeeze(dim=0).float()

    logprobs = gather_logprobs(logits, torch.roll(packed_input_ids, shifts=-1))
    logprobs = torch.where(prompt_mask, 0, logprobs)

    loss = -logprobs.sum() / prompt_mask.logical_not().count_nonzero()

    with torch.no_grad():
        seqlogp = torch.zeros(
            cu_seqlens.shape[0] - 1, device=logits.device, dtype=torch.float64
        )
        for i in range(cu_seqlens.shape[0] - 1):
            m = prompt_mask[cu_seqlens[i] - i : cu_seqlens[i + 1] - i - 1]
            logp = logprobs[cu_seqlens[i] - i : cu_seqlens[i + 1] - i - 1]
            assert cu_seqlens[i + 1] - i - 1 <= logprobs.shape[0], (
                cu_seqlens,
                logprobs.shape,
            )
            seqlogp[i] = torch.where(m, 0.0, logp.detach()).sum() / (
                m.numel() - m.count_nonzero()
            )

    ## Loggin stats
    stats_tracker.denominator(
        n_seqs=torch.ones(
            cu_seqlens.shape[0] - 1, dtype=torch.bool, device=logprobs.device
        ),
        n_tokens=torch.ones(logits.shape[0], dtype=torch.bool, device=logits.device),
        n_valid_tokens=prompt_mask.logical_not(),
        prompt_tokens=prompt_mask,
    )
    stats_tracker.stat(ppl=(-seqlogp).exp().float(), denominator="n_seqs")
    stats_tracker.stat(loss=-logprobs.detach(), denominator="n_valid_tokens")
    vocab_min_logits = logits.detach().min(-1).values.float()
    vocab_max_logits = logits.detach().max(-1).values.float()
    stats_tracker.stat(
        vocab_min_logits=vocab_min_logits,
        vocab_max_logits=vocab_max_logits,
        denominator="n_tokens",
    )

    return loss


class SFTTrainer(Trainer):

    def __init__(
        self,
        args: TrainingArgs,
        trainer_config: TrainerConfig,
        train_dataset: Dataset,
        valid_dataset: Optional[Dataset] = None,
        rollout_controller: Optional[RolloutController] = None,
    ):
        super().__init__(
            args,
            trainer_config,
            train_dataset,
            valid_dataset,
            rollout_controller,
        )

        self.config = config = trainer_config.sft
        assert config is not None

        engine_factory = EngineFactory(args)
        self.model = engine_factory.make_engine(config.model)
        self.tokenizer = load_hf_tokenizer(config.model.path)
        # self.processor=None
        # if self.model.engine_config.type._class in VALID_VISION_MODELS:
        #     self.processor,self.tokenizer = load_hf_processor_and_tokenizer(
        #         config.model.path,
        #     )

        self.mb_spec = config.mb_spec

        self.save_ctl = timeutil.EpochStepTimeFreqCtl(
            freq_epoch=self.args.exp_ctrl.save_freq_epochs,
            freq_step=self.args.exp_ctrl.save_freq_steps,
            freq_sec=self.args.exp_ctrl.save_freq_secs,
        )
        self.eval_ctl = timeutil.EpochStepTimeFreqCtl(
            freq_epoch=self.args.exp_ctrl.eval_freq_epochs,
            freq_step=self.args.exp_ctrl.eval_freq_steps,
            freq_sec=self.args.exp_ctrl.eval_freq_steps,
        )
        self.summary_writer = init_stats_logging(args)

    def _tokenize(self, strs: List[str]):
        # tokenize strings into unpadded tokens with lengths.
        return self.tokenizer(
            strs,
            padding=False,
            truncation=True,
            return_length=True,
            max_length=self.mb_spec.max_tokens_per_mb,
            return_attention_mask=False,
        )
    # def _process(self,images):
    #     assert self.processor is not None, "Processor is not initialized for vision model"
    #     # image_list=[]
    #     breakpoint()
    #     # for image in images:
    #     #     image_list.append(process_image(image))

    #     return self.processor(
    #         images,
    #         return_tensors="pt",
    #         padding=True,
    #         truncation=True,
    #     )

    def _get_packed_input(self, data: Dict):

        prompts = data["prompt"]
        answers = data["answer"]
        inputs = [
            prompt + answer + self.tokenizer.eos_token
            for prompt, answer in zip(prompts, answers)
        ]
        tokenized_prompts = self._tokenize(prompts)
        tokenized_inputs = self._tokenize(inputs)

        # form a data batch
        prompt_lens = tokenized_prompts["length"]
        input_lens = tokenized_inputs["length"]

        input_lens = torch.tensor(input_lens, dtype=torch.int)
        input_ids = [
            torch.tensor(seq, dtype=torch.long) for seq in tokenized_inputs["input_ids"]
        ]

        prompt_mask = []
        for input_len, prompt_len in zip(input_lens, prompt_lens):
            assert input_len >= prompt_len, (input_len, prompt_len)
            pm = [1] * prompt_len + [0] * (input_len - prompt_len)
            prompt_mask.append(torch.tensor(pm, dtype=torch.bool))

        cu_seqlens = torch.nn.functional.pad(
            input_lens.cumsum(0, dtype=torch.int), (1, 0)
        )
        max_seqlen = int(torch.max(input_lens).item())
        packed_input_ids = torch.cat(input_ids, dim=0)
        prompt_mask = torch.cat(prompt_mask, dim=0)
        total_seqlen = int(cu_seqlens[-1].item())
        position_ids = compute_varlen_position_indices(total_seqlen, cu_seqlens)

        return dict(
            input_ids=packed_input_ids.unsqueeze(0).cuda(),
            attention_mask=None,
            position_ids=position_ids.unsqueeze(0).cuda(),
            prompt_mask=prompt_mask.unsqueeze(0).cuda(),
            cu_seqlens=cu_seqlens.cuda(),
            max_seqlen=max_seqlen,
            use_cache=False,
        )

    def _get_packed_vl_input(self, data: Dict):
        # breakpoint()
        vl_prompt_input_ids= data["vl_prompt_input_ids"]
        vl_prompt_length = data["vl_prompt_length"]
        answer_input_ids = data["answer_input_ids"]
        answer_length = data["answer_length"]
        eos_token_tensor = torch.tensor([self.tokenizer.eos_token_id], dtype=torch.long)
        # merge vl_prompt_input_ids, answer_input_ids, adding eos token,the first column is batch size
        tokenized_inputs = {
            "input_ids": [
                torch.cat(
                    [
                        vl_prompt_input_ids[i],
                        answer_input_ids[i],
                        eos_token_tensor,
                    ],
                    dim=0,
                )
                for i in range(len(vl_prompt_input_ids))
            ],
            "length": [vl_prompt_length[i] + answer_length[i] + 1 for i in range(len(vl_prompt_length))],
        }

        # breakpoint()
        pixel_values = data["pixel_values"]
        image_grid_thw = data.get("image_grid_thw", None)
        # form a data batch
        prompt_lens = vl_prompt_length
        input_lens = tokenized_inputs["length"]

        input_lens = torch.tensor(input_lens, dtype=torch.int)
        input_ids = [
            torch.tensor(seq, dtype=torch.long) for seq in tokenized_inputs["input_ids"]
        ]

        prompt_mask = []
        for input_len, prompt_len in zip(input_lens, prompt_lens):
            assert input_len >= prompt_len, (input_len, prompt_len)
            pm = [1] * prompt_len + [0] * (input_len - prompt_len)
            prompt_mask.append(torch.tensor(pm, dtype=torch.bool))

        cu_seqlens = torch.nn.functional.pad(
            input_lens.cumsum(0, dtype=torch.int), (1, 0)
        )
        max_seqlen = int(torch.max(input_lens).item())
        packed_input_ids = torch.cat(input_ids, dim=0)
        prompt_mask = torch.cat(prompt_mask, dim=0)
        total_seqlen = int(cu_seqlens[-1].item())
        position_ids = compute_varlen_position_indices(total_seqlen, cu_seqlens)
        # breakpoint()
        return dict(
            input_ids=packed_input_ids.unsqueeze(0).cuda(),
            attention_mask=None,
            position_ids=position_ids.unsqueeze(0).cuda(),
            prompt_mask=prompt_mask.unsqueeze(0).cuda(),
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            cu_seqlens=cu_seqlens.cuda(),
            max_seqlen=max_seqlen,
            use_cache=False,
        )


    def train(self, resume_from_checkpoint=None):
        self.create_train_dataloader()

        total_epochs = self.args.exp_ctrl.total_train_epochs
        steps_per_epoch = len(self.train_dataloader)
        ft_spec = FinetuneSpec(
            total_train_epochs=steps_per_epoch,
            dataset_size=len(self.train_dataset),
            train_batch_size=self.args.train_dataset.batch_size,
        )

        self.model.init_distributed(None, ft_spec)
        self.model.train()

        if dist.get_rank() == 0:
            print(f"total_epochs={total_epochs} step_per_epoch={steps_per_epoch}")
        global_step = 0
        start_time = time.monotonic()
        # dataloader: self.train_data_loader

        for epoch in range(total_epochs):
            self.data_generator = iter(self.train_dataloader)
            for step in range(steps_per_epoch):
                timing_stats = {}
                with record_timing("timeperf/data_processing", timing_stats):
                    data = next(self.data_generator)
                    if self.model.engine_config.type._class in VALID_VISION_MODELS:
                        packed_input_data = self._get_packed_vl_input(data)
                    else:
                        packed_input_data = self._get_packed_input(data)
                    dist.barrier()

                with record_timing("timeperf/train_step", timing_stats):
                    with stats_tracker.scope("sft"):
                        stats = self.model.train_batch(
                            input_=packed_input_data,
                            loss_fn=compute_packed_sft_loss,
                            loss_weight_fn=lambda x: x["prompt_mask"]
                            .logical_not()
                            .count_nonzero(),
                            mb_spec=self.mb_spec,
                        )
                        self.model.step_lr_scheduler()
                        stats_tracker.scalar(**stats)

                if self.save_ctl.check(
                    epochs=int(step == steps_per_epoch - 1), steps=1
                ):
                    if dist.get_rank() == 0:
                        print("Saving model ...")

                    with record_timing("timeperf/save", timing_stats):
                        save_path = get_save_checkpoint_path(
                            self.args, epoch, step, global_step
                        )
                        self.model.save_model_to_hf(save_path, self.tokenizer)

                if self.eval_ctl.check(
                    epochs=int(step == steps_per_epoch - 1), steps=1
                ):
                    if dist.get_rank() == 0:
                        print("Running evaluation ...")
                    with record_timing("timeperf/eval", timing_stats):
                        self._eval(global_step)

                training_stats = stats_tracker.export()
                training_stats.update(timing_stats)
                log_wandb_tensorboard(global_step, training_stats, self.summary_writer)

                if dist.get_rank() == 0:
                    print(
                        f"Epoch {epoch} Step {step} GlobalStep {global_step} done. Detailed time stats:"
                        f"\n{tabulate_stats(timing_stats, floatfmt='.2f')}"
                    )
                global_step += 1

        if dist.get_rank() == 0:
            print(
                f"Training completes! Total time elapsed {time.monotonic() - start_time:.2f}."
            )

        close_wandb_tensorboard(self.summary_writer)

    def _eval(self, global_step):
        self.create_valid_dataloader()
        if self.valid_dataloader is None:
            return

        self.eval_data_generator = iter(self.valid_dataloader)
        n_steps = len(self.valid_dataloader)

        losses = []

        start_time = time.monotonic()
        for step in range(n_steps):
            data = next(self.eval_data_generator)
            packed_input_data = self._get_packed_input(data)
            with stats_tracker.scope("sft-eval"):
                avg_loss = self.model.eval_batch(
                    input_=packed_input_data,
                    loss_fn=compute_packed_sft_loss,
                    loss_weight_fn=lambda x: x["prompt_mask"]
                    .logical_not()
                    .count_nonzero(),
                    mb_spec=self.mb_spec,
                )
                losses.append(avg_loss)
        val_loss = torch.mean(torch.stack(losses))

        print(
            f"Global step: {global_step} evaluation time cost {time.monotonic() - start_time:.2f} "
            f"val_loss={val_loss:.4f}"
        )
