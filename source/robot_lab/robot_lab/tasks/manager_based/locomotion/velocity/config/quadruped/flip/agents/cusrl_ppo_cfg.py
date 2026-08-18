# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import cusrl
from cusrl.environment.isaaclab import TrainerCfg


@dataclass
class FlipRoughTrainerBaseCfg(TrainerCfg):
    max_iterations = 20000
    save_interval = 100
    experiment_name = "flip_base_rough"
    agent_factory = cusrl.ActorCritic.Factory(
        num_steps_per_update=24,
        actor_factory=cusrl.Actor.Factory(
            backbone_factory=cusrl.Mlp.Factory(
                hidden_dims=[512, 256, 128], activation_fn="ELU", ends_with_activation=True
            ),
            distribution_factory=cusrl.NormalDist.Factory(),
        ),
        critic_factory=cusrl.Value.Factory(
            backbone_factory=cusrl.Mlp.Factory(
                hidden_dims=[512, 256, 128], activation_fn="ELU", ends_with_activation=True
            ),
        ),
        optimizer_factory=cusrl.OptimizerFactory("AdamW", defaults={"lr": 1.0e-3}),
        sampler=cusrl.AutoMiniBatchSampler(num_epochs=5, num_mini_batches=4),
        hooks=[
            cusrl.hook.ValueComputation(),
            cusrl.hook.GeneralizedAdvantageEstimation(gamma=0.99, lamda=0.95),
            cusrl.hook.AdvantageNormalization(),
            cusrl.hook.ValueLoss(),
            cusrl.hook.OnPolicyPreparation(),
            cusrl.hook.PpoSurrogateLoss(),
            cusrl.hook.EntropyLoss(weight=0.008),
            cusrl.hook.GradientClipping(max_grad_norm=1.0),
            cusrl.hook.OnPolicyStatistics(sampler=cusrl.AutoMiniBatchSampler()),
            cusrl.hook.AdaptiveLRSchedule(desired_kl_divergence=0.01),
        ],
    )


@dataclass
class FlipFERKRoughTrainerCfg(FlipRoughTrainerBaseCfg):
    experiment_name = "flip_FERK_rough"


@dataclass
class FlipFERKFlatTrainerCfg(FlipFERKRoughTrainerCfg):
    max_iterations = 5000
    experiment_name = "flip_FERK_flat"


@dataclass
class FlipFERERoughTrainerCfg(FlipRoughTrainerBaseCfg):
    experiment_name = "flip_FERE_rough"


@dataclass
class FlipFEREFlatTrainerCfg(FlipFERERoughTrainerCfg):
    max_iterations = 5000
    experiment_name = "flip_FERE_flat"


@dataclass
class FlipFKRKRoughTrainerCfg(FlipRoughTrainerBaseCfg):
    experiment_name = "flip_FKRK_rough"


@dataclass
class FlipFKRKFlatTrainerCfg(FlipFKRKRoughTrainerCfg):
    max_iterations = 5000
    experiment_name = "flip_FKRK_flat"


@dataclass
class FlipFKRERoughTrainerCfg(FlipRoughTrainerBaseCfg):
    experiment_name = "flip_FKRE_rough"


@dataclass
class FlipFKREFlatTrainerCfg(FlipFKRERoughTrainerCfg):
    max_iterations = 5000
    experiment_name = "flip_FKRE_flat"
