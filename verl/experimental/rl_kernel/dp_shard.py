# SPDX-License-Identifier: Apache-2.0
"""Lockstep data-parallel sharding for the torchrun phase actors.

The phase actors run on every rank with no controller, so verl does not split
data across DP for us. When the MoE all-to-all spans the DP ranks of a pipeline
stage (EP>1), every DP rank must execute the same number of calls with the same
microbatch count. These pure helpers assign cases to DP ranks, pad uneven
shards with discardable duplicates, and restore the original case order from the
output-owning ranks. With ``dp_size == 1`` they are the identity.

Assignment is strided: DP rank ``d`` owns original indices ``d, d + dp, d + 2dp, ...``.
Every rank gets ``ceil(n / dp)`` slots. A missing slot is padded with a duplicate
of the rank's last real case (or case 0 when the rank owns none); its output is
dropped and it must never contribute loss or advantages.
"""


def shard_indices(n, dp_rank, dp_size):
    """Original case index per local slot; ``None`` marks a padding slot."""
    if n <= 0 or dp_size <= 0 or not 0 <= dp_rank < dp_size:
        raise ValueError(f"invalid shard request: {n=} {dp_rank=} {dp_size=}")
    rounds = -(-n // dp_size)
    return [dp_rank + slot * dp_size if dp_rank + slot * dp_size < n else None for slot in range(rounds)]


def shard_cases(cases, dp_rank, dp_size):
    """Return (local cases padded to ceil(n/dp), original indices with None for padding)."""
    indices = shard_indices(len(cases), dp_rank, dp_size)
    local = []
    for index in indices:
        if index is not None:
            local.append(cases[index])
        else:
            local.append(local[-1] if local else cases[0])
    return local, indices


def require_even(n, dp_size, what="training batch"):
    """Training never pads: a padded duplicate would contribute loss."""
    if n % dp_size:
        raise ValueError(f"{what} of {n} samples is not divisible by data-parallel size {dp_size}")


def merge_gathered(gathered, n, dp_size):
    """Restore original case order from ``all_gather_object`` payloads.

    ``gathered`` has one entry per world rank: ``None`` for ranks that own no
    outputs, else ``(dp_rank, results)`` with results aligned to that rank's
    local slots. Padding slots are dropped. Exactly one output owner per DP rank
    is required, and every original case must be covered exactly once.
    """
    owners = {}
    for payload in gathered:
        if payload is None:
            continue
        dp_rank, results = payload
        if dp_rank in owners:
            raise RuntimeError(f"data-parallel rank {dp_rank} has more than one output-owning rank")
        owners[dp_rank] = results
    if set(owners) != set(range(dp_size)):
        raise RuntimeError(f"output owners cover DP ranks {sorted(owners)}, expected {dp_size}")
    ordered = [None] * n
    for dp_rank, results in owners.items():
        indices = shard_indices(n, dp_rank, dp_size)
        if len(results) != len(indices):
            raise RuntimeError("local result count differs from the lockstep slot count")
        for index, value in zip(indices, results, strict=True):
            if index is not None:
                ordered[index] = value
    if any(value is None for value in ordered):
        raise RuntimeError("gathered results do not cover every case")
    return ordered


def gather_in_order(local_results, n, dp_rank, dp_size, is_output_rank):
    """All-gather per-slot results over WORLD and return all ``n`` results in case order.

    Only ranks with ``is_output_rank`` (``engine.is_mp_src_rank_with_outputs()``)
    contribute; ``local_results`` is ignored elsewhere. Every rank returns the
    full ordered list.
    """
    import torch.distributed as dist

    payload = (dp_rank, list(local_results)) if is_output_rank else None
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, payload)
    return merge_gathered(gathered, n, dp_size)
