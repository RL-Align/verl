# Copyright 2026 Individual Contributor
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Standalone replica extension used by verl's normal LLMServerManager."""

import os

import ray
from rl_engine.integrations.orchestrators.verl.pipeline_rollout import RLKPipelineVLLMHttpServer

from verl.workers.rollout.replica import RolloutMode
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMReplica


class RLKTrainerVLLMServer(RLKPipelineVLLMHttpServer):
    def __init__(self, **kwargs):
        visible = kwargs["cuda_visible_devices"]
        os.environ["HIP_VISIBLE_DEVICES"] = visible
        os.environ["CUDA_VISIBLE_DEVICES"] = visible
        reservations = kwargs.pop("workers")
        if kwargs["rollout_mode"] != RolloutMode.STANDALONE or len(reservations) != kwargs["gpus_per_node"]:
            raise ValueError("expected one reservation per local standalone GPU")
        # Match native MCore's original checkpoint ownership (layers 0-2 / 3).
        # These settings affect this dedicated server and its mp children only.
        os.environ["RLK_NODE_ID"] = ray.get_runtime_context().get_node_id()
        os.environ["RLK_SYNC_DIR"] = str(kwargs["config"].rlk_sync_dir or "/outputs/policy-sync")
        os.environ.update(
            VLLM_PP_LAYER_PARTITION=os.environ.get("RLK_ROLLOUT_PP_SPLIT") or os.environ.get("RLK_DSV4_PP_SPLIT", "3,1"),
            VLLM_ROCM_USE_AITER="1",
            VLLM_ENABLE_V1_MULTIPROCESSING="1",
        )
        # These are resource holders, not model workers. The native vLLM mp
        # executor constructs the actual PP stages on their reserved devices.
        super().__init__(workers=[], **kwargs)
        from .runtime import process_identity

        self._rlk_source_identity = process_identity("rollout-ray-server")

    def _preprocess_engine_kwargs(self, engine_kwargs):
        super()._preprocess_engine_kwargs(engine_kwargs)
        engine_kwargs["worker_cls"] = "verl.experimental.rl_kernel.rollout_worker.RLKVerlDSV4Worker"
        os.environ["RLK_VERL_MHC_BACKEND"] = self.config.engine_kwargs["vllm"].get("rlk_mhc_backend", "rocm_triton")
        engine_kwargs.pop("rlk_mhc_backend", None)

    async def rlk_placement(self):
        return await self.collective_rpc("rlk_placement")

    async def rlk_load_policy(self, path, global_steps):
        if not self._submission_paused:
            raise RuntimeError("close admission and clear old caches before policy sync")
        acks = await self.collective_rpc("rlk_sync_weights", args=(path,))
        self.global_steps = global_steps
        return acks

    async def rlk_verify_current_policy(self, path):
        if self._submission_paused or self.engine.output_processor.request_states:
            raise RuntimeError("final readback requires an admitted, idle replica")
        return await self.collective_rpc("rlk_readback_policy", args=(path,))

    async def rlk_score_sequences(self, cases, label):
        """Score fixed responses by native prompt-logprob prefill, with real B2."""
        import asyncio
        import time
        from uuid import uuid4

        if self._submission_paused or self.engine.output_processor.request_states:
            raise RuntimeError("audit requires an admitted, idle current-version replica")
        await self.engine.pause_generation(mode="keep", clear_cache=True)
        request_ids = [f"rlk-{label}-{i}-{uuid4().hex}" for i in range(len(cases))]
        await self.collective_rpc("rlk_begin_audit", args=(request_ids, label))
        tasks = [
            asyncio.create_task(
                self.generate(
                    request_id=request,
                    prompt_ids=case["prompt_ids"] + case["response_ids"],
                    sampling_params=dict(
                        temperature=1.0,
                        top_p=1.0,
                        top_k=-1,
                        logprobs=True,
                        prompt_logprobs=0,
                        max_tokens=1,
                    ),
                )
            )
            for request, case in zip(request_ids, cases, strict=True)
        ]
        deadline = time.monotonic() + 60
        while len(self.engine.output_processor.request_states) != len(cases):
            for task in tasks:
                if task.done():
                    task.result()
                    raise RuntimeError("audit request finished before paused batch admission")
            if time.monotonic() > deadline:
                raise RuntimeError("native batch admission timed out")
            await asyncio.sleep(0.01)
        await self.engine.resume_generation()
        outputs = await asyncio.gather(*tasks)
        stages = await self.collective_rpc("rlk_end_audit")
        scores = []
        for case, output in zip(cases, outputs, strict=True):
            start = len(case["prompt_ids"]) - 1
            stop = start + len(case["response_ids"])
            ids = output.extra_fields["prompt_ids"][start:stop]
            if [row[0] for row in ids] != case["response_ids"]:
                raise RuntimeError("native prompt logprob causal shift/token IDs differ")
            scores.append([row[0] for row in output.extra_fields["prompt_logprobs"][start:stop]])
        for stage in stages:
            if len(stage["batches"]) != 1 or stage["batches"][0]["num_requests"] != len(cases):
                raise RuntimeError(f"audit was split instead of real B{len(cases)}: {stage}")
            if len(cases) > 1 and not stage["order"]:
                raise RuntimeError("actual requested AB/BA order was not observed")
        return dict(scores=scores, stages=stages, global_steps=self.global_steps)


class RLKTrainerReplica(vLLMReplica):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.server_class = ray.remote(RLKTrainerVLLMServer)

    async def init_standalone(self, resource_pool=None):
        if self.config.rlk_node_ips:
            from .cluster import pinned_pool

            if resource_pool is not None:
                raise ValueError("explicit placement cannot also take an external pool")
            resource_pool = pinned_pool(list(self.config.rlk_node_ips), self.gpus_per_replica_node, "rlk-rollout", 2)
        await super().init_standalone(resource_pool)
