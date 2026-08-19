from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
	from isaaclab.envs import ManagerBasedRLEnv


class StructureCommand(CommandTerm):
	"""Command term that selects and smoothly transitions between robot structures."""

	cfg: StructureCommandCfg

	def __init__(self, cfg: StructureCommandCfg, env: ManagerBasedRLEnv):
		super().__init__(cfg, env)

		if not cfg.structure_list:
			raise ValueError("structure_list must contain at least one structure")

		self.asset: Articulation = env.scene[cfg.asset_name]
		self._joint_ids, _ = self.asset.find_joints(cfg.joint_names)
		if not self._joint_ids:
			raise ValueError(f"No joints found matching pattern '{cfg.joint_names}'")

		default_joint_pos = self.asset.data.default_joint_pos[0, self._joint_ids].clone()
		structure_positions = []
		for structure in cfg.structure_list:
			joint_pos = default_joint_pos.clone()
			for joint_pattern, position in structure.items():
				joint_ids, _ = self.asset.find_joints(joint_pattern)
				if not joint_ids:
					raise ValueError(f"No joints found matching pattern '{joint_pattern}'")
				local_ids = [self._joint_ids.index(joint_id) for joint_id in joint_ids]
				joint_pos[local_ids] = position
			structure_positions.append(joint_pos)

		self.structure_positions = torch.stack(structure_positions).to(self.device)
		self.command_buffer = torch.zeros(self.num_envs, 1, dtype=torch.int32, device=self.device)
		self.current_joint_pos = default_joint_pos.expand(self.num_envs, -1).clone()
		self.previous_joint_pos = self.current_joint_pos.clone()
		self.target_joint_pos = self.current_joint_pos.clone()
		self.transition_alpha = torch.ones(self.num_envs, device=self.device)
		self._last_structure_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

	@property
	def command(self) -> torch.Tensor:
		"""Return structure IDs with shape ``(num_envs, 1)``."""
		return self.command_buffer

	def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> dict[str, float]:
		"""Sample and immediately apply one structure for the reset environments."""
		if env_ids is None:
			reset_ids = torch.arange(self.num_envs, device=self.device)
		else:
			reset_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

		self._resample_command(reset_ids)
		self.current_joint_pos[reset_ids] = self.target_joint_pos[reset_ids]
		self.previous_joint_pos[reset_ids] = self.target_joint_pos[reset_ids]
		self.transition_alpha[reset_ids] = 1.0
		self._last_structure_ids[reset_ids] = self.command_buffer[reset_ids, 0].to(torch.long)

		default_joint_vel = self.asset.data.default_joint_vel[reset_ids, self._joint_ids].clone()
		self.asset.write_joint_state_to_sim(
			self.target_joint_pos[reset_ids], default_joint_vel, joint_ids=self._joint_ids, env_ids=reset_ids
		)
		return {}

	def _resample_command(self, env_ids: Sequence[int]):
		"""Sample structure IDs for the selected environments."""
		env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
		sampled_ids = torch.randint(len(self.structure_positions), (len(env_ids),), device=self.device)
		self.command_buffer[env_ids, 0] = sampled_ids.to(torch.int32)

	def _update_command(self):
		"""Detect command changes and advance all active structure transitions."""
		structure_ids = self.command_buffer[:, 0].to(torch.long)
		changed = structure_ids != self._last_structure_ids
		if changed.any():
			changed_ids = torch.where(changed)[0]
			self.previous_joint_pos[changed_ids] = self.current_joint_pos[changed_ids]
			self.target_joint_pos[changed_ids] = self.structure_positions[structure_ids[changed_ids]]
			self.transition_alpha[changed_ids] = 0.0
			self._last_structure_ids[changed_ids] = structure_ids[changed_ids]

		active = self.transition_alpha < 1.0
		if not active.any():
			return

		active_ids = torch.where(active)[0]
		dt = self._env.cfg.sim.dt * self._env.cfg.decimation
		self.transition_alpha[active_ids] = torch.clamp(
			self.transition_alpha[active_ids] + dt / self.cfg.transition_duration, max=1.0
		)
		alpha = self.transition_alpha[active_ids].unsqueeze(1)
		smooth_alpha = alpha * alpha * (3.0 - 2.0 * alpha)
		next_joint_pos = (
			(1.0 - smooth_alpha) * self.previous_joint_pos[active_ids]
			+ smooth_alpha * self.target_joint_pos[active_ids]
		)
		joint_vel = (next_joint_pos - self.current_joint_pos[active_ids]) / dt
		self.current_joint_pos[active_ids] = next_joint_pos
		self.asset.write_joint_state_to_sim(
			next_joint_pos, joint_vel, joint_ids=self._joint_ids, env_ids=active_ids
		)


@configclass
class StructureCommandCfg(CommandTermCfg):
	"""Configuration for a structure command with smooth runtime transitions."""

	class_type: type = StructureCommand

	asset_name: str = "robot"
	joint_names: str = ".*"
	structure_list: list[dict[str, float]] = []
	transition_duration: float = 0.3
