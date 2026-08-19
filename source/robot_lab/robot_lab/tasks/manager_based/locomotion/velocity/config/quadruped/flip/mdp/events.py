
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class StructureResetManager:

    _instance: StructureResetManager | None = None

    def __init__(self) -> None:
        self._frames: dict[str, dict[str, torch.Tensor]] = {}
        self._structure_pos_list: dict[int, torch.Tensor] = {}
        self._structure_ids: list[int] = []

    @classmethod
    def get(cls) -> StructureResetManager:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def init(
        self,
        env: ManagerBasedEnv,
        device: str | torch.device,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        scturture_list: list[dict[str, float]] | None = None,
    ) -> None:

        asset: Articulation = env.scene[asset_cfg.name]

        # get default joint state
        single_structure_pos = asset.data.default_joint_pos[0].clone().to(device)
        self._structure_pos_list = {}
        self._structure_ids = []

        if scturture_list is None:
            self._structure_pos_list[0] = single_structure_pos
            self._structure_ids.append(0)
        else:
            for structure_id, structure_pos in enumerate(scturture_list):
                joint_pos = single_structure_pos.clone()
                for joint_name_pattern, position in structure_pos.items():
                    joint_ids, _ = asset.find_joints(joint_name_pattern)
                    if len(joint_ids) == 0:
                        raise ValueError(f"No joints found matching pattern '{joint_name_pattern}'")
                    joint_pos[joint_ids] = position

                self._structure_pos_list[structure_id] = joint_pos
                self._structure_ids.append(structure_id)
       
    def reset(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        structure_command: str = "structure_id",
    ) -> torch.Tensor:
        return self.get_structure_joint_pos(env, env_ids, asset_cfg, structure_command)

    def get_structure_ids(self)-> list[int]:
        return self._structure_ids

    def get_structure_joint_pos(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        structure_command: str = "structure_id",
    ) -> torch.Tensor:
        """Return joint positions selected by the structure command."""
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device)
        else:
            env_ids = env_ids.to(device=env.device, dtype=torch.long)

        command_term = env.command_manager.get_term(structure_command)
        structure_ids = command_term.command[env_ids].reshape(-1).to(torch.long)
        available_ids = torch.tensor(self._structure_ids, device=env.device)
        if not torch.all(torch.isin(structure_ids, available_ids)):
            raise ValueError(f"Unknown structure ID in command: {structure_ids.tolist()}")

        joint_pos = torch.stack(
            [self._structure_pos_list[int(structure_id)] for structure_id in structure_ids]
        ).to(env.device)
        return joint_pos[:, asset_cfg.joint_ids]

    
#环境重置的时候重置构型
def sturcture_reset_joints_by_scale(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    structure_command: str = "structure_id",
):
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    env_ids = env_ids.to(device=env.device, dtype=torch.long)

    # cast env_ids to allow broadcasting
    if not isinstance(asset_cfg.joint_ids, slice):
        iter_env_ids = env_ids[:, None]
    else:
        iter_env_ids = env_ids

    # get default joint state
    joint_pos = StructureResetManager.get().reset(
        env, env_ids, asset_cfg, structure_command=structure_command
    )
    joint_vel = asset.data.default_joint_vel[iter_env_ids, asset_cfg.joint_ids].clone()

    # scale these values randomly
    joint_pos *= math_utils.sample_uniform(*position_range, joint_pos.shape, joint_pos.device)
    joint_vel *= math_utils.sample_uniform(*velocity_range, joint_vel.shape, joint_vel.device)

    # clamp joint pos to limits
    joint_pos_limits = asset.data.soft_joint_pos_limits[iter_env_ids, asset_cfg.joint_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    # clamp joint vel to limits
    joint_vel_limits = asset.data.soft_joint_vel_limits[iter_env_ids, asset_cfg.joint_ids]
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

    # set into the physics simulation
    asset.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)

def sturcture_init(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    structure_list: list[dict[str, float]] | None = None,
) -> None:
    """Initialize the available robot structures during the startup event."""
    del env_ids
    StructureResetManager.get().init(
        env=env,
        device=env.device,
        asset_cfg=asset_cfg,
        scturture_list=structure_list,
    )

def get_structure_ids() -> list[int]:
    """Get the available robot structure IDs."""
    return StructureResetManager.get().get_structure_ids()