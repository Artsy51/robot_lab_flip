
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from .events import StructureResetManager

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
        

def strucutre_joint_deviation_l1(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    structure_command: str = "structure_id",
) -> torch.Tensor:
    """Penalize joint positions that deviate from the commanded structure."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    target_joint_pos = StructureResetManager.get().get_structure_joint_pos(
        env=env,
        env_ids=None,
        asset_cfg=asset_cfg,
        structure_command=structure_command,
    )
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - target_joint_pos
    return torch.sum(torch.abs(angle), dim=1)

