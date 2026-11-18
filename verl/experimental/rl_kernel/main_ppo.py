# isort: skip_file
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
# ruff: noqa: E402 -- configure native import ABI before importing verl workers.
"""Run the normal verl v1 GRPO pipeline with RLK's DSV4 engine/rollout glue."""

from .runtime import prepare_worker

prepare_worker()

from . import trainer_engine  # noqa: F401

import hydra
import ray
from omegaconf import OmegaConf
from verl.trainer.main_ppo import run_ppo
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.logging_utils import configure_verl_logging

from .trainer import RLKVerlTrainer
from .trainer_contract import validate_trainer_config


@ray.remote
class RLKTaskRunner:
    def run(self, config):
        import transfer_queue as tq
        from verl.trainer.ppo.v1 import AgentLoopManagerTQ

        configure_verl_logging()
        validate_trainer_config(config)
        config.transfer_queue.enable = True
        OmegaConf.resolve(config)
        from .cluster import preflight

        preflight(config)
        tq.init(config.transfer_queue)
        trainer = None
        succeeded = False
        try:
            trainer = RLKVerlTrainer(config)
            trainer.init()
            fqn = config.actor_rollout_ref.rollout.agent.get("agent_loop_manager_class")
            manager_cls = load_class_from_fqn(fqn, "AgentLoopManager") if fqn else AgentLoopManagerTQ
            agents = manager_cls.create(
                config=config,
                llm_client=trainer.get_llm_client(),
                teacher_client=trainer.get_teacher_client(),
                reward_loop_worker_handles=trainer.get_reward_handles(),
            )
            trainer.fit(agents)
            succeeded = True
        finally:
            try:
                if trainer is not None:
                    trainer.close_dataloaders()
            finally:
                try:
                    logger = getattr(trainer, "logger", None)
                    if logger is not None:
                        logger.finish(exit_code=0 if succeeded else 1)
                finally:
                    tq.close()


@hydra.main(config_path="config", config_name="dsv4_grpo", version_base=None)
def main(config):
    import json
    from pathlib import Path

    from .runtime import process_identity

    Path(config.rlk.sync_dir).mkdir(parents=True, exist_ok=True)
    Path(config.rlk.sync_dir, "driver-identity.json").write_text(json.dumps(process_identity("driver"), indent=2))
    auto_set_device(config)
    validate_trainer_config(config)
    validate_config(config, use_reference_policy=False, use_critic=False)
    # vLLM's driver import creates a CUDA alias. Do not pass that four-device
    # alias into the Ray head: Ray assigns HIP visibility per GPU actor later.
    # A worker setup hook is too early, before Ray applies its accelerator IDs.
    import os

    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    run_ppo(config, task_runner_class=RLKTaskRunner)


if __name__ == "__main__":
    main()
