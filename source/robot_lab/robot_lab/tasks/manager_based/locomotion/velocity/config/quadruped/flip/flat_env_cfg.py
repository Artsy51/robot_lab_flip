# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from .rough_env_cfg import (
    FlipFERERoughEnvCfg,
    FlipFERKRoughEnvCfg,
    FlipFKRERoughEnvCfg,
    FlipFKRKRoughEnvCfg,
)


def _configure_flat_env(cfg):
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
class FlipFERKFlatEnvCfg(FlipFERKRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_flat_env(self)


@configclass
class FlipFEREFlatEnvCfg(FlipFERERoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_flat_env(self)


@configclass
class FlipFKRKFlatEnvCfg(FlipFKRKRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_flat_env(self)


@configclass
class FlipFKREFlatEnvCfg(FlipFKRERoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_flat_env(self)
