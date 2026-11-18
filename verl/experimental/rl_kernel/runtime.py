# SPDX-License-Identifier: Apache-2.0
"""Prepare a Ray process before deserializing GPU-owning actor classes."""

import os
import socket
from uuid import uuid4

_PROCESS_INSTANCE = {}


def prepare_worker():
    # Ray's AMD accelerator manager assigns HIP visibility. vLLM imports can
    # otherwise see CUDA visibility inherited from the four-device driver.
    # This hook runs before actor deserialization, never expands the assignment.
    hip = os.environ.get("HIP_VISIBLE_DEVICES")
    if hip is not None:
        if os.environ.get("ROCR_VISIBLE_DEVICES"):
            raise RuntimeError("Use one HIP visibility namespace for the isolated Ray job")
        os.environ["CUDA_VISIBLE_DEVICES"] = hip


def process_identity(role):
    """Record source identity in the actual driver, Ray actor or mp worker."""
    import hashlib
    import importlib.metadata as metadata
    import subprocess
    import sys
    from pathlib import Path

    import torch

    projects = {}
    for name, module, distribution in [
        ("verl", "verl", "verl"),
        ("RL-Kernel", "rl_engine", "rl-kernel"),
        ("vllm", "vllm", "vllm"),
        ("Megatron-LM", "megatron.core", "megatron-core"),
    ]:
        root = Path("/opt/dsv4") / name

        def git(*args, _root=root):
            return subprocess.check_output(["git", "-C", str(_root), *args], text=True).strip()

        loaded = sys.modules.get(module)
        projects[name] = dict(
            sha=git("rev-parse", "HEAD"),
            url=git("config", "--get", "remote.origin.url"),
            import_path=getattr(loaded, "__file__", None),
            dirty=bool(git("status", "--porcelain")),
            diff_sha256=hashlib.sha256(
                subprocess.check_output(["git", "-C", str(root), "diff", "--binary", "HEAD"])
            ).hexdigest(),
            version=metadata.version(distribution),
        )
    dependencies = {}
    for name in ("megatron-bridge", "torch", "triton", "amd-aiter", "ray", "TransferQueue"):
        dependencies[name] = metadata.version(name)
    ray_ids = None
    import ray

    if ray.is_initialized():
        ray_ids = ray.get_runtime_context().get_accelerator_ids()
    identity = dict(
        role=role,
        pid=os.getpid(),
        hostname=socket.gethostname(),
        node_id=(
            ray.get_runtime_context().get_node_id()
            if ray.is_initialized()
            else os.environ.get("RLK_NODE_ID", socket.gethostname())
        ),
        process_instance=_PROCESS_INSTANCE.setdefault(os.getpid(), uuid4().hex),
        projects=projects,
        dependencies=dependencies,
        hip=torch.version.hip,
        ray_accelerator_ids=ray_ids,
        visibility={
            k: os.environ.get(k) for k in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")
        },
    )
    print("RLK_PROCESS_IDENTITY", identity, flush=True)
    return identity
