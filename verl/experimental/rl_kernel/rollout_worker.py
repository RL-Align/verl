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
"""DSV4 worker for normal verl generation, without parity-harness trace hooks."""

import os
from contextlib import ExitStack

import torch
from vllm.v1.worker.gpu_worker import Worker


class RLKVerlDSV4Worker(Worker):
    def load_model(self, **kwargs):
        from rl_engine.integrations.engines.rollout.vllm.mi35x_moe import native_w4a16_loading

        with native_w4a16_loading():
            super().load_model(**kwargs)
        from rl_engine.integrations.common.device_identity import hip_device_identity
        from rl_engine.integrations.engines.rollout.vllm.group_projection import (
            bind_original_group_projection,
        )
        from rl_engine.integrations.engines.rollout.vllm.native_decode import bind_rollout_decode
        from rl_engine.integrations.engines.rollout.vllm.native_sparse import bind_rollout_sparse
        from rl_engine.integrations.engines.rollout.vllm.outputs import (
            bind_output_head,
            selected_raw_logprobs,
        )
        from rl_engine.integrations.engines.rollout.vllm.router_projection import (
            bind_rollout_router_projection,
        )
        from rl_engine.integrations.engines.rollout.vllm.shared_down import bind_shared_down
        from rl_engine.integrations.orchestrators.verl.mhc import bind_mhc_worker
        from vllm.distributed import get_pp_group

        from .runtime import process_identity

        self._rlk_source_identity = process_identity("vllm-mp-worker")
        identity = self._rlk_source_identity
        self._rlk_process_key = f"{identity['node_id']}:{identity['pid']}:{identity['process_instance']}"
        wrapper = self.model_runner.get_model()
        model = wrapper.model
        self._rlk_source_identity["actual_model_class"] = type(wrapper).__module__ + "." + type(wrapper).__qualname__
        self._rlk_source_identity["actual_model_source"] = __import__("inspect").getfile(type(wrapper))
        self._rlk_pp_rank = get_pp_group().rank_in_group
        self._rlk_device_identity = hip_device_identity()
        self._rlk_worker = bind_mhc_worker(
            model,
            "rollout",
            backend=os.environ.get("RLK_VERL_MHC_BACKEND", "rocm_triton"),
            mode="split",
            version=0,
        )
        bind_shared_down(model, self._rlk_worker)
        for index in range(model.start_layer, model.end_layer):
            layer = model.layers[index]
            from rl_engine.integrations.engines.rollout.vllm.compressor_projection import (
                bind_rollout_compressor_projection,
            )

            bind_rollout_compressor_projection(layer.attn)
            bind_rollout_router_projection(layer.ffn.gate)
            bind_rollout_sparse(layer.attn)
            bind_original_group_projection(layer.attn, self.model_config.model, index)
            bind_rollout_decode(layer.attn)
        from rl_engine.integrations.engines.rollout.vllm.batch_order import bind_prefill_order

        self._rlk_audit_order = []
        self._rlk_audit_batches = None
        bind_prefill_order(self.model_runner, lambda: self._rlk_audit_order)
        original = model.forward

        def forward(*args, **kwargs):
            if self._rlk_audit_batches is not None:
                from vllm.forward_context import get_forward_context

                metadata = get_forward_context().attn_metadata
                batch = next(m for m in metadata.values() if hasattr(m, "num_reqs"))
                starts = batch.query_start_loc.detach().cpu().tolist()
                ids = kwargs.get("input_ids", args[0] if args else None)
                tokens = ids.detach().cpu().tolist() if ids is not None else None
                self._rlk_audit_batches.append(
                    dict(
                        num_requests=int(batch.num_reqs),
                        query_start_loc=starts,
                        input_token_rows=(
                            None
                            if tokens is None
                            else [tokens[a:b] for a, b in zip(starts[:-1], starts[1:], strict=True)]
                        ),
                    )
                )
            with self._rlk_worker.request(self._rlk_worker.version):
                return original(*args, **kwargs)

        model.forward = forward
        if get_pp_group().is_last_rank:
            bind_output_head(wrapper, self._rlk_worker)
            if self.model_runner.sampler.logprobs_mode != "raw_logprobs":
                raise ValueError("RLK verl rollout requires raw_logprobs")
            from vllm.v1.worker.gpu.sample import logprob

            # The vLLM mp executor owns this process. Match its selected-token
            # provider ABI; sampling and token ordering remain owned by vLLM.
            def raw(logits, token_ids):
                return selected_raw_logprobs(wrapper, logits, token_ids, weight_version=self._rlk_worker.version)

            logprob.compute_token_logprobs = raw

    def rlk_sync_weights(self, path):
        from rl_engine.integrations.engines.rollout.vllm.checkpoint_sync import (
            pipeline_stage_payload,
            readback_original_tensor,
            verify_loaded_update,
        )

        wrapper = self.model_runner.get_model()
        payload = pipeline_stage_payload(wrapper, torch.load(path, map_location="cuda", weights_only=True))
        worker = self._rlk_worker
        version = payload["weight_version"]
        if version < worker.version or (version == worker.version and version != 0):
            raise ValueError("rollout weight version must advance after initial verification")
        with ExitStack() as stack:
            if version == 0:
                # Both engines loaded the same original checkpoint. Verify it
                # without fabricating an optimizer update or a version advance.
                stack.enter_context(worker.request(0))
                loaded = {
                    readback_original_tensor(wrapper, name, tensor)[0] for name, tensor in payload["weights"].items()
                }
            else:
                stack.enter_context(worker.updating(version, torch.cuda.synchronize))
                loaded = wrapper.load_weights(payload["weights"].items())
            checked = verify_loaded_update(wrapper, payload, loaded)
        return dict(
            source_identity=self._rlk_source_identity,
            pipeline_stage=self._rlk_pp_rank,
            pid=os.getpid(),
            worker_identity=self._rlk_process_key,
            node_id=self._rlk_source_identity["node_id"],
            device_identity=self._rlk_device_identity,
            original_layer_ids=list(range(wrapper.model.start_layer, wrapper.model.end_layer)),
            weight_version=worker.version,
            parameter_load_completed=True,
            verified_original_tensors=checked,
        )

    def rlk_placement(self):
        return dict(
            rank=self._rlk_pp_rank,
            node_id=self._rlk_source_identity["node_id"],
            pid=os.getpid(),
            worker_identity=self._rlk_process_key,
            **self._rlk_device_identity,
        )

    def rlk_readback_policy(self, path):
        """Verify current bytes after stress without loading or advancing a version."""
        from rl_engine.integrations.engines.rollout.vllm.checkpoint_sync import (
            pipeline_stage_payload,
            readback_original_tensor,
            verify_loaded_update,
        )

        wrapper = self.model_runner.get_model()
        payload = pipeline_stage_payload(wrapper, torch.load(path, map_location="cuda", weights_only=True))
        if payload["weight_version"] != self._rlk_worker.version:
            raise RuntimeError("readback requested a stale policy")
        with self._rlk_worker.request(payload["weight_version"]):
            names = {readback_original_tensor(wrapper, name, tensor)[0] for name, tensor in payload["weights"].items()}
            checked = verify_loaded_update(wrapper, payload, names)
        return dict(
            pid=os.getpid(),
            worker_identity=self._rlk_process_key,
            node_id=self._rlk_source_identity["node_id"],
            pipeline_stage=self._rlk_pp_rank,
            weight_version=self._rlk_worker.version,
            readback_only=True,
            verified_original_tensors=checked,
        )

    def rlk_begin_audit(self, request_ids, label):
        import re

        pattern = os.environ.get("RLK_TRACE_AUDIT_PATTERN")
        matched = pattern and re.fullmatch(pattern, label)
        first = os.environ.get("RLK_TRACE_FIRST_AUDIT") == "1" and not getattr(self, "_rlk_traced", False)
        if matched or first or label.startswith("failure-"):
            from examples.operators.dsv4_4l.boundary_trace import capture

            model = self.model_runner.get_model().model
            model._rlk_trace_worker = self._rlk_worker
            self._rlk_stop_trace = capture(
                model,
                f"{os.environ.get('RLK_SYNC_DIR', '/outputs/policy-sync')}/traces/"
                f"rollout-rank-{self._rlk_pp_rank}-{label}",
                "rollout",
            )
            self._rlk_traced = True
        self._rlk_audit_order = list(request_ids)
        self._rlk_audit_batches = []
        self.model_runner._rlk_prefill_order_evidence = []
        return dict(pid=os.getpid(), weight_version=self._rlk_worker.version)

    def rlk_end_audit(self):
        if getattr(self, "_rlk_stop_trace", None):
            self._rlk_stop_trace()
            self._rlk_stop_trace = None
        result = dict(
            pid=os.getpid(),
            worker_identity=self._rlk_process_key,
            node_id=self._rlk_source_identity["node_id"],
            pipeline_stage=self._rlk_pp_rank,
            weight_version=self._rlk_worker.version,
            batches=self._rlk_audit_batches,
            order=self.model_runner._rlk_prefill_order_evidence,
        )
        self._rlk_audit_order = []
        self._rlk_audit_batches = None
        return result
