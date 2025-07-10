import torch


@torch.compile
def gather_logprobs(logits: torch.Tensor, labels: torch.Tensor):
    log_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return log_probs_labels


from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist


@torch.compile
@torch.no_grad()
def calc_entropy(logits, cu_seqlens):
    probs = torch.nn.functional.softmax(logits.detach().float(), dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-7), dim=-1)
    return entropy


@torch.no_grad()
def masked_normalization(
    x: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dim=None,
    unbiased=False,
    eps=1e-5,
    high_precision=True,
    all_reduce=True,
    reduce_group=None,
):
    dtype = torch.float64 if high_precision else torch.float32
    x = x.to(dtype)
    if dim is None:
        dim = tuple(range(len(x.shape)))
    if mask is None:
        factor = torch.tensor(
            np.prod([x.shape[d] for d in dim]), dtype=dtype, device=x.device
        )
    else:
        mask = mask.to(dtype)
        x = x * mask
        factor = mask.sum(dim, keepdim=True)
    x_sum = x.sum(dim=dim, keepdim=True)
    x_sum_sq = x.square().sum(dim=dim, keepdim=True)
    if dist.is_initialized() and all_reduce:
        dist.all_reduce(factor, op=dist.ReduceOp.SUM, group=reduce_group)
        dist.all_reduce(x_sum, op=dist.ReduceOp.SUM, group=reduce_group)
        dist.all_reduce(
            x_sum_sq,
            op=dist.ReduceOp.SUM,
            group=reduce_group,
        )
    mean = x_sum / factor
    meansq = x_sum_sq / factor
    var = meansq - mean**2
    if unbiased:
        var *= factor / (factor - 1)
    return ((x - mean) / (var.sqrt() + eps)).float()


def ppo_actor_loss_fn(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    eps_clip: float,
    loss_mask: torch.Tensor,
    c_clip: Optional[float] = None,
    proximal_logprobs: Optional[torch.Tensor] = None,
    behav_imp_weight_cap: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict]:
    denorm_logprobs = (
        proximal_logprobs if proximal_logprobs is not None else old_logprobs
    )
    loss_mask_count = loss_mask.count_nonzero() or 1
    ratio = torch.where(loss_mask, torch.exp(logprobs - denorm_logprobs), 0)
    clipped_ratio = torch.clamp(ratio, 1.0 - eps_clip, 1.0 + eps_clip)
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * clipped_ratio
    clip_mask = pg_loss1.detach() < pg_loss2.detach()
    pg_loss = torch.max(pg_loss1, pg_loss2)
    if c_clip is not None:
        assert c_clip > 1.0, c_clip
        pg_loss3 = torch.sign(advantages) * c_clip * advantages
        dual_clip_mask = pg_loss3.detach() < pg_loss.detach()
        pg_loss = torch.min(pg_loss, pg_loss3)
    else:
        dual_clip_mask = torch.zeros_like(clip_mask)
    if proximal_logprobs is not None:
        behav_kl = proximal_logprobs - old_logprobs
        behav_imp_weight = behav_kl.exp()
        behav_mask = (
            (behav_imp_weight <= behav_imp_weight_cap).logical_and(loss_mask)
            if behav_imp_weight_cap is not None
            else loss_mask
        )
        behav_kl = torch.where(behav_mask, behav_kl, 0.0)
        behav_imp_weight = torch.where(behav_mask, behav_imp_weight, 0.0)
        pg_loss = pg_loss * behav_imp_weight
    logging_loss = pg_loss.detach()
    pg_loss = torch.where(loss_mask, pg_loss, 0).sum() / loss_mask_count
    clip_mask.logical_and_(loss_mask)
    dual_clip_mask.logical_and_(loss_mask)
    stat = dict(
        loss=logging_loss,
        importance_weight=ratio.detach(),
        approx_kl=(logprobs - denorm_logprobs).detach(),
        clip_mask=clip_mask,
        dual_clip_mask=dual_clip_mask,
    )
    if proximal_logprobs is not None:
        stat["behave_imp_weight"] = behav_imp_weight
        stat["behave_approx_kl"] = behav_kl
        stat["behave_mask"] = behav_mask
    return pg_loss, stat
