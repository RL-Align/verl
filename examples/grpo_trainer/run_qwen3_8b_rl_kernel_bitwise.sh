#!/usr/bin/env bash
# Three matched GRPO steps per arm on one 8xH100 node (TP4/CP2, vLLM TP4).
set -euo pipefail

VERL_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
WORKSPACE_ROOT=$(cd -- "${VERL_ROOT}/.." && pwd)
RL_KERNEL_ROOT=${RL_KERNEL_ROOT:-"${WORKSPACE_ROOT}/RL-Kernel"}
MODEL_ROOT=${MODEL_ROOT:-"${WORKSPACE_ROOT}/artifacts/models/Qwen3-8B"}
DATA_ROOT=${DATA_ROOT:-"${WORKSPACE_ROOT}/artifacts/data/gsm8k"}
RUN_ROOT=${RUN_ROOT:-"${WORKSPACE_ROOT}/artifacts/runs/bitwise-$(date -u +%Y%m%dT%H%M%SZ)"}
PYTHON=${PYTHON:-"${VERL_ROOT}/.venv/bin/python"}
ARM=${1:-both}
MATCHED_GRAPHS=${MATCHED_GRAPHS:-0}
MATCHED_COMPILE_MODE=${MATCHED_COMPILE_MODE:-3}

if [[ "${MATCHED_GRAPHS}" != 0 && "${MATCHED_GRAPHS}" != 1 ]]; then
  echo "MATCHED_GRAPHS must be 0 or 1" >&2
  exit 2
fi
if [[ "${MATCHED_COMPILE_MODE}" != 0 && "${MATCHED_COMPILE_MODE}" != 3 ]]; then
  echo "MATCHED_COMPILE_MODE must be 0 or 3" >&2
  exit 2
fi

if [[ "${ARM}" != native && "${ARM}" != native_det && "${ARM}" != rl_kernel && "${ARM}" != both ]]; then
  echo "Usage: $0 [native|native_det|rl_kernel|both]" >&2
  exit 2
fi
if [[ ! -f "${MODEL_ROOT}/config.json" || ! -f "${DATA_ROOT}/train.parquet" || ! -f "${DATA_ROOT}/test.parquet" ]]; then
  echo "Qwen3-8B weights and GSM8K parquet data are required" >&2
  exit 2
fi
if [[ ! -x "${PYTHON}" ]]; then
  echo "Missing Python environment: ${PYTHON}" >&2
  exit 2
fi

export PATH="$(dirname "${PYTHON}"):${PATH}"
export PYTHONPATH="${VERL_ROOT}:${RL_KERNEL_ROOT}:${PYTHONPATH:-}"
export VERL_USE_UV=0
export VERL_LOGP_BITWISE_AUDIT=1
export WANDB_MODE=disabled
export NCCL_NVLS_ENABLE=0
export RL_KERNEL_STRICT_CANONICAL_TP=4
export RL_KERNEL_STRICT_CANONICAL_VOCAB_SIZE=152064
export RL_KERNEL_VLLM_REAL_VOCAB_SIZE=151936
export RL_KERNEL_VLLM_PADDED_VOCAB_SIZE=152064
# Keep the established numerical contract by default; allow explicit CUDA tile experiments.
export RL_KERNEL_LOGPROB_NUM_VOCAB_TILES=${RL_KERNEL_LOGPROB_NUM_VOCAB_TILES:-64}

run_arm() (
  local arm=$1
  local output="${RUN_ROOT}/${arm}"
  local -a rollout_overrides=()
  if [[ "${MATCHED_GRAPHS}" == 1 ]]; then
    export VLLM_USE_V2_MODEL_RUNNER=0
    rollout_overrides=("++actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config={mode:${MATCHED_COMPILE_MODE},cudagraph_mode:FULL_DECODE_ONLY,cudagraph_capture_sizes:[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]}")
  fi
  if [[ -e "${output}" ]]; then
    echo "Refusing to overwrite existing results: ${output}" >&2
    exit 2
  fi
  mkdir -p "${output}"
  export VERL_LOGP_AUDIT_DIR="${output}"
  if [[ "${arm}" == rl_kernel ]]; then
    export VERL_RL_KERNEL=1
    export VIME_RL_KERNEL_STRICT=1
    export RL_KERNEL_VLLM_INTEGRATION=1
    export VLLM_PLUGINS=rl_kernel
    export VLLM_ATTENTION_BACKEND=FLASH_ATTN
    export RL_KERNEL_ATTENTION_CASE=R/R RL_KERNEL_FFN_CASE=R/R RL_KERNEL_LOGP_CASE=R/R
    export RL_KERNEL_CANONICAL_CP_GRAD=1
    export CUBLAS_WORKSPACE_CONFIG=:16:8
    export CUBLASLT_WORKSPACE_SIZE=1
    export RL_KERNEL_DET_GEMM_BACKEND=cublaslt_nosplitk
    export RL_KERNEL_CUDA_ONLY=1
    # Only the strict CUDA arm opts in; the shared RL-Kernel default stays unchanged.
    export RL_KERNEL_CUDA_GRAPH_MULTIBLOCK_MIN_BYTES=${RL_KERNEL_CUDA_GRAPH_MULTIBLOCK_MIN_BYTES:-32768}
    export VLLM_BATCH_INVARIANT=1
    export RL_KERNEL_READBACK_DIR="${output}/readbacks"
    export RL_KERNEL_VLLM_CUDAGRAPH_MAX_CAPTURE_SIZE=16
    # Same settings as the vime consistency arm. RL-Kernel routes CUDA rollout
    # logp through the V1 sampler, so keep vLLM >= 0.27 off Model Runner V2.
    export VLLM_USE_V2_MODEL_RUNNER=0
    export NCCL_ALGO=Ring
    export RL_KERNEL_COMPLETE_SAMPLING_SUPPORT=1
    export RL_KERNEL_VLLM_TEMPERATURE=1.0
    export RL_KERNEL_SEED=1234 RL_KERNEL_ROLLOUT_SEED=1234
    if [[ "${MATCHED_GRAPHS}" == 0 ]]; then
      rollout_overrides=('++actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config={mode:0,cudagraph_mode:NONE}')
    fi
    "${PYTHON}" -c 'from importlib.metadata import entry_points; assert any(ep.name == "rl_kernel" for ep in entry_points(group="vllm.general_plugins")), "RL-Kernel vLLM plugin is not installed"'
    "${PYTHON}" -c 'from rl_engine.kernels.ops.base import _C; assert all(hasattr(_C, name) for name in ("det_gemm_fwd", "linear_logp_local_bf16_forward")), "RL-Kernel SM90 CUDA extension is incomplete"'
  else
    unset VERL_RL_KERNEL VIME_RL_KERNEL_STRICT RL_KERNEL_VLLM_INTEGRATION
    unset VLLM_PLUGINS
    unset VLLM_ATTENTION_BACKEND
    unset RL_KERNEL_ATTENTION_CASE RL_KERNEL_FFN_CASE RL_KERNEL_LOGP_CASE RL_KERNEL_CANONICAL_CP_GRAD
    unset NCCL_ALGO RL_KERNEL_COMPLETE_SAMPLING_SUPPORT RL_KERNEL_VLLM_TEMPERATURE
    if [[ "${MATCHED_GRAPHS}" == 0 ]]; then
      unset VLLM_USE_V2_MODEL_RUNNER
    fi
    if [[ "${arm}" == native_det ]]; then
      # verl's documented full-determinism switches (docs/advance/determinism.md).
      rollout_overrides=(
        actor_rollout_ref.rollout.full_determinism=true
        # vLLM torch.compile autotuning is rejected by deterministic Inductor.
        actor_rollout_ref.rollout.enforce_eager=true
        ++actor_rollout_ref.actor.megatron.full_determinism=true
        ++actor_rollout_ref.ref.megatron.full_determinism=true
      )
    fi
  fi
  if [[ "${PROFILE_ROLLOUT:-0}" == 1 ]]; then
    rollout_overrides+=(
      global_profiler.tool=torch 'global_profiler.steps=[2]'
      "global_profiler.save_path=${output}/profile"
      actor_rollout_ref.rollout.profiler.enable=true
      'actor_rollout_ref.rollout.profiler.ranks=[0]'
      'actor_rollout_ref.rollout.profiler.tool_config.torch.contents=[cuda]'
      actor_rollout_ref.rollout.profiler.tool_config.torch.discrete=true
      actor_rollout_ref.rollout.profiler.tool_config.torch.profile_token_start=64
      actor_rollout_ref.rollout.profiler.tool_config.torch.profile_token_end=80
    )
  fi

  cd "${VERL_ROOT}"
  MODEL_PATH="${MODEL_ROOT}" TRAIN_BATCH_SIZE=8 PPO_MINI_BATCH_SIZE=8 \
    MAX_PROMPT_LENGTH=256 MAX_RESPONSE_LENGTH=1024 PPO_MAX_TOKEN_LEN_PER_GPU=4096 \
    ACTOR_TP=4 ACTOR_PP=1 ROLLOUT_TP=4 ROLLOUT_N=4 \
    ROLLOUT_GPU_MEM_UTIL=0.4 SAVE_FREQ=-1 TEST_FREQ=-1 \
    EXPERIMENT_NAME="qwen3_8b_${arm}_bitwise" \
    bash examples/grpo_trainer/run_qwen3_8b_megatron.sh \
    data.train_files="${DATA_ROOT}/train.parquet" \
    data.val_files="${DATA_ROOT}/test.parquet" \
    actor_rollout_ref.actor.megatron.context_parallel_size=2 \
    actor_rollout_ref.ref.megatron.context_parallel_size=2 \
    actor_rollout_ref.actor.megatron.sequence_parallel=false \
    actor_rollout_ref.ref.megatron.sequence_parallel=false \
    ++actor_rollout_ref.actor.megatron.override_transformer_config.sequence_parallel=false \
    ++actor_rollout_ref.ref.megatron.override_transformer_config.sequence_parallel=false \
    ++actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=flash \
    ++actor_rollout_ref.ref.megatron.override_transformer_config.attention_backend=flash \
    actor_rollout_ref.actor.megatron.vanilla_mbridge=True \
    actor_rollout_ref.ref.megatron.vanilla_mbridge=True \
    actor_rollout_ref.actor.megatron.seed=1234 \
    actor_rollout_ref.ref.megatron.seed=1234 \
    actor_rollout_ref.rollout.seed=1234 \
    trainer.total_training_steps=3 trainer.val_before_train=False \
    'trainer.logger=[console]' trainer.resume_mode=disable trainer.use_v1=False \
    trainer.default_local_dir="${output}/checkpoints" \
    ray_kwargs.ray_init.runtime_env.py_executable="${PYTHON}" \
    "${rollout_overrides[@]}" \
    >"${output}/train.log" 2>&1
  if [[ "${MATCHED_GRAPHS}" == 1 ]]; then
    if ! grep -Fq "\"mode\": ${MATCHED_COMPILE_MODE}" "${output}/train.log" || \
       ! grep -Fq '"cudagraph_mode": "FULL_DECODE_ONLY"' "${output}/train.log" || \
       [[ $(grep -Fc 'Capturing CUDA graphs (decode, FULL): 100%' "${output}/train.log") -lt 2 ]]; then
      echo "Matched vLLM compilation or decode CUDA Graph capture was not observed" >&2
      exit 1
    fi
  fi
  "${PYTHON}" - "${output}" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
rows = [json.loads(line) for line in (run_dir / "mismatch.jsonl").read_text().splitlines()]
if [row["step"] for row in rows] != [1, 2, 3]:
    raise SystemExit(f"expected exactly steps 1, 2, 3 in {run_dir}")
if any(row["training/rollout_logp_compared"] <= 0 for row in rows):
    raise SystemExit("each step must compare active tokens")
summary = {
    "arm": run_dir.name,
    "steps": rows,
    "last_mismatch_count": rows[-1]["training/rollout_logp_mismatch_count"],
    "last_max_abs_diff": rows[-1]["training/rollout_logp_max_abs_diff"],
}
(run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
if run_dir.name == "rl_kernel" and any(
    row["training/rollout_logp_mismatch_count"] != 0 or row["training/rollout_logp_max_abs_diff"] != 0
    for row in rows
):
    raise SystemExit("strict RL-Kernel run was not bitwise aligned")
PY
)

if [[ "${ARM}" == both || "${ARM}" == native ]]; then
  run_arm native
fi
if [[ "${ARM}" == native_det ]]; then
  run_arm native_det
fi
if [[ "${ARM}" == both || "${ARM}" == rl_kernel ]]; then
  run_arm rl_kernel
fi
echo "Results: ${RUN_ROOT}"
