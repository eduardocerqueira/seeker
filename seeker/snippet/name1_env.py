#date: 2025-10-28T17:03:28Z
#url: https://api.github.com/gists/b65fc9e9f1e06ce16d28b08a8536765a
#owner: https://api.github.com/users/jmake

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnv
from isaaclab.assets import Articulation
from isaaclab.utils.math import sample_uniform
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from .name1_env_cfg import Name1EnvCfg


def CreateYaws(num_envs) : 
    commands = torch.randn((num_envs, 3)).cuda()
    commands[:,-1] = 0.0
    commands = commands / torch.linalg.norm(commands, dim=1, keepdim=True)

    ratio = commands[:,1] / (commands[:,0] + 1E-8)
    gzero = torch.where(commands > 0, True, False)
    lzero = torch.where(commands < 0, True, False)

    plus = lzero[:,0] * gzero[:,1]
    minus = lzero[:,0] * lzero[:,1]
    offsets = torch.pi * plus - torch.pi * minus

    yaws = torch.zeros( (num_envs,1) ).cuda()
    yaws = torch.atan(ratio).reshape(-1,1) + offsets.reshape(-1,1)
    return commands,yaws 


def ResetIdx(robot, env_ids: Sequence[int] | None, base) :
    if env_ids is None: 
        env_ids = robot._ALL_INDICES

    base.super()._reset_idx(env_ids)

    ## CreateYaws -> 
    commands[env_ids] = torch.randn((len(env_ids), 3)).cuda()
    commands[env_ids,-1] = 0.0
    commands[env_ids] = commands[env_ids] / torch.linalg.norm(commands[env_ids], dim=1, keepdim=True)
    
    ratio = commands[env_ids][:,1] / (commands[env_ids][:,0]+1E-8)
    gzero = torch.where(commands[env_ids] > 0, True, False)
    lzero = torch.where(commands[env_ids]< 0, True, False)
    plus = lzero[:,0]*gzero[:,1]
    minus = lzero[:,0]*lzero[:,1]
    offsets = torch.pi*plus - torch.pi*minus
    yaws[env_ids] = torch.atan(ratio).reshape(-1,1) + offsets.reshape(-1,1)
    ## <- CreateYaws 

    ## Reseting 
    default_root_state = robot.data.default_root_state[env_ids]
    default_root_state[:,:3] += scene.env_origins[env_ids]

    robot.write_root_state_to_sim(default_root_state, env_ids)
    return commands,yaws 


def SetupScene(cfg, scene) :
    robot = Articulation(cfg.robot_cfg)

    spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
    scene.clone_environments(copy_from_source=False)
    scene.articulations["robot"] = robot

    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)
    return robot


def ApplyAction(robot, actions, indices) : 
    robot.set_joint_velocity_target(actions, joint_ids=indices)
    return 


def GetObservations(robot) :
    ## Linear and angular velocities of the 
    ## articulation root’s center of mass frame relative to the world
    ## root_com_vel_w -> [lin_vel, ang_vel] 

    ## assets/articulation/articulation_data.py
    ## ArticulationData -> root_com_vel_w 
    velocity = robot.data.root_com_vel_w 

    ## assets/rigid_object_collection/rigid_object_collection_data.py
    ## RigidObjectCollectionData -> FORWARD_VEC_B / GRAVITY_VEC_W
    ## 
    ##  'x direction' (FORWARD_VEC_B = [1,0,0]) is the 'forward direction' for the asset
    ## 
    forwards = math_utils.quat_apply(robot.data.root_link_quat_w, robot.data.FORWARD_VEC_B)

    dot = torch.sum(forwards * commands, dim=-1, keepdim=True)
    cross = torch.cross(forwards, commands, dim=-1)[:,-1].reshape(-1,1)

    ## ?? 
    ## root_com_lin_vel_w -> Root center of mass linear velocity in SIMULATION WORLD frame.
    ## root_com_lin_vel_b -> Root center of mass linear velocity in             BASE frame (BODY frame?!!)
    ## ?? 
    forward_speed = robot.data.root_com_lin_vel_b[:,0].reshape(-1,1)

    obs = torch.hstack((dot, cross, forward_speed))

    ## Improvement, keep the observation space as small as possible, use : 
    # obs = torch.hstack((self.velocity, self.commands))
    ## and, in 'IsaacLabTutorialEnvCfg' -> observation space back to 3 
    return  {"policy": obs}


def GetRewards(robot, forwards, commands) -> torch.Tensor :
    ## 
    ##  'x direction' is the 'forward direction' for the asset
    ##
    ## 'x component' of the 'linear center of mass velocity' of the robot in the 'body frame'
    ## this should be equivalent to 'inner product' between 
    ## the 'forward vector' and the 'linear velocity' in the 'world frame'  
    ##
    forward_reward = robot.data.root_com_lin_vel_b[:,0].reshape(-1,1)

    ## 'alignment', 'inner product' between the 'forward vector' and the 'command vector'
    ##     same direction, alignment_reward ->  1
    ## opposite direction, alignment_reward -> -1
    alignment_reward = torch.sum(forwards * commands, dim=-1, keepdim=True) # <- forwards, GetObservations

    ## Driving at 'full speed' in the 'direction of the command'
    total_reward = forward_reward + alignment_reward
    return total_reward ## To be maximized!!

    ## Speed up!!
    ## - Large negative values to be near zero
    ## - Agent can be reward for driving forward OR being aligned to the command but
    ##   logical AND suggests multiplication 
    #total_reward = forward_reward * torch.exp(alignment_reward)
    #return total_reward


class Name1Env(DirectRLEnv) : cfg: Name1EnvCfg

    def _pre_physics_step(self, actions: torch.Tensor) -> None :
        self.actions = actions.clone()
        return 

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor] :
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return False, time_out


    def __init__(self, cfg: Name1EnvCfg, render_mode: str | None = None, **kwargs) :
        super().__init__(cfg, render_mode, **kwargs)

        ## X.0. 
        self.dof_idx, _ = self.robot.find_joints(self.cfg.dof_names)
        self.joint_pos  = self.robot.data.joint_pos
        self.joint_vel  = self.robot.data.joint_vel
        return 


    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)


    def _apply_action(self) -> None:
        self.robot.set_joint_effort_target(self.actions * self.cfg.action_scale, joint_ids=self._cart_dof_idx)

    def _get_observations(self) -> dict:
        obs = torch.cat(
            (
                self.joint_pos[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
                self.joint_vel[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
                self.joint_pos[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
                self.joint_vel[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
            ),
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        total_reward = compute_rewards(
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_terminated,
            self.cfg.rew_scale_pole_pos,
            self.cfg.rew_scale_cart_vel,
            self.cfg.rew_scale_pole_vel,
            self.joint_pos[:, self._pole_dof_idx[0]],
            self.joint_vel[:, self._pole_dof_idx[0]],
            self.joint_pos[:, self._cart_dof_idx[0]],
            self.joint_vel[:, self._cart_dof_idx[0]],
            self.reset_terminated,
        )
        return total_reward

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_pos[:, self._pole_dof_idx] += sample_uniform(
            self.cfg.initial_pole_angle_range[0] * math.pi,
            self.cfg.initial_pole_angle_range[1] * math.pi,
            joint_pos[:, self._pole_dof_idx].shape,
            joint_pos.device,
        )
        joint_vel = self.robot.data.default_joint_vel[env_ids]

        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


@torch.jit.script
def compute_rewards(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_pole_pos: float,
    rew_scale_cart_vel: float,
    rew_scale_pole_vel: float,
    pole_pos: torch.Tensor,
    pole_vel: torch.Tensor,
    cart_pos: torch.Tensor,
    cart_vel: torch.Tensor,
    reset_terminated: torch.Tensor,
):
    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
    rew_termination = rew_scale_terminated * reset_terminated.float()
    rew_pole_pos = rew_scale_pole_pos * torch.sum(torch.square(pole_pos).unsqueeze(dim=1), dim=-1)
    rew_cart_vel = rew_scale_cart_vel * torch.sum(torch.abs(cart_vel).unsqueeze(dim=1), dim=-1)
    rew_pole_vel = rew_scale_pole_vel * torch.sum(torch.abs(pole_vel).unsqueeze(dim=1), dim=-1)
    total_reward = rew_alive + rew_termination + rew_pole_pos + rew_cart_vel + rew_pole_vel
    return total_reward