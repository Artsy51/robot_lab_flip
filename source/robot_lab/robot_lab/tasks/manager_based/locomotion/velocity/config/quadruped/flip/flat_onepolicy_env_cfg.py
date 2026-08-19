# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from .rough_onepolicy_env_cfg import FlipFERERoughEnvCfg


def _configure_flat_env(cfg: FlipFERERoughEnvCfg) -> None:
    cfg.rewards.base_height_l2.params["sensor_cfg"] = None
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    cfg.scene.height_scanner = None
    cfg.scene.height_scanner_base = None
    cfg.observations.policy.height_scan = None
    cfg.observations.critic.height_scan = None
    cfg.curriculum.terrain_levels = None
    cfg.disable_zero_weight_rewards()


@configclass
class FlipFlatOnePolicyEnvCfg(FlipFERERoughEnvCfg):
    """Flat-terrain one-policy environment with commanded robot structures."""

    def __post_init__(self) -> None:
        super().__post_init__()
        _configure_flat_env(self)
