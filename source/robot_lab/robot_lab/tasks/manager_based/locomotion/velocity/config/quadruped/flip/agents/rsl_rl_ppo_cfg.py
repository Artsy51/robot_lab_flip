# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class FlipRoughPPORunnerBaseCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 100
    experiment_name = "flip_base_rough"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class FlipFERKRoughPPORunnerCfg(FlipRoughPPORunnerBaseCfg):
    experiment_name = "flip_FERK_rough"


@configclass
class FlipFERKFlatPPORunnerCfg(FlipFERKRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 8000
        self.experiment_name = "flip_FERK_flat"


@configclass
class FlipFERERoughPPORunnerCfg(FlipRoughPPORunnerBaseCfg):
    experiment_name = "flip_FERE_rough"


@configclass
class FlipFEREFlatPPORunnerCfg(FlipFERERoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 8000
        self.experiment_name = "flip_FERE_flat"


@configclass
class FlipFKRKRoughPPORunnerCfg(FlipRoughPPORunnerBaseCfg):
    experiment_name = "flip_FKRK_rough"


@configclass
class FlipFKRKFlatPPORunnerCfg(FlipFKRKRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 8000
        self.experiment_name = "flip_FKRK_flat"


@configclass
class FlipFKRERoughPPORunnerCfg(FlipRoughPPORunnerBaseCfg):
    experiment_name = "flip_FKRE_rough"


@configclass
class FlipFKREFlatPPORunnerCfg(FlipFKRERoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 14000
        self.experiment_name = "flip_FKRE_flat"
