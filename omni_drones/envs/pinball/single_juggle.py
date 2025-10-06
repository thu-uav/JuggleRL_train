# MIT License
#
# Copyright (c) 2023 Botian Xu, Tsinghua University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from typing import Optional
import functorch

import omni.isaac.core.utils.torch as torch_utils
import omni_drones.utils.kit as kit_utils
from omni_drones.utils.torch import euler_to_quaternion, quat_axis, quaternion_to_euler
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.core.objects as objects
import omni.isaac.core.materials as materials
import torch
import torch.distributions as D

from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.robots.drone import MultirotorBase
from omni_drones.views import RigidPrimView
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import (
    UnboundedContinuousTensorSpec, CompositeSpec, DiscreteTensorSpec
)
from pxr import UsdShade, PhysxSchema
import logging
from carb import Float3
from omni.isaac.debug_draw import _debug_draw
from .common import rectangular_cuboid_edges,_carb_float3_add
from omni_drones.utils.torchrl.transforms import append_to_h5
from omni.isaac.orbit.sensors import ContactSensorCfg, ContactSensor
from omegaconf import DictConfig
import os



def quat_rotate(q, v):
    """
    Rotate vector v by quaternion q
    q: quaternion [w, x, y, z]
    v: vector [x, y, z]
    Returns the rotated vector
    """

    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    
    # Quaternion rotation formula
    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z
    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z
    
    # Construct the rotation matrix from the quaternion
    rot_matrix = torch.stack([
        ww + xx - yy - zz,
        2 * (xy - wz),
        2 * (xz + wy),
        
        2 * (xy + wz),
        ww - xx + yy - zz,
        2 * (yz - wx),
        
        2 * (xz - wy),
        2 * (yz + wx),
        ww - xx - yy + zz
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)

    v_expanded = v.expand(*q.shape[:-1], 3)
    
    # Rotate the vector using the rotation matrix
    return torch.matmul(rot_matrix, v_expanded.unsqueeze(-1)).squeeze(-1)

class SingleJuggle(IsaacEnv):
    def __init__(self, cfg, headless):
        self.use_eval = cfg.task.use_eval
        self.num_drones = 1
        self.ball_mass: float = cfg.task.ball_mass
        self.ball_radius: float = cfg.task.ball_radius
        self.base_restitution = cfg.task.get("base_restitution", 0.8) 

        # 添加弹性系数随机化参数
        self.use_restitution_randomization = cfg.task.get("use_restitution_randomization", False)  # 是否使用弹性系数随机化
        self.restitution_range = cfg.task.get("restitution_range", [0.7, 0.85])  # 弹性系数的范围
        
        # 添加击球间隔参数
        self.true_hit_step_gap = cfg.task.get("true_hit_step_gap", 35)  # 两次有效击球之间的最小步数间隔
        
        super().__init__(cfg, headless)

        self.reward_action_smoothness_weight: float = cfg.task.reward_action_smoothness_weight
        self.time_encoding = self.cfg.task.time_encoding
        self.on_the_spot = cfg.task.on_the_spot
        self.hit_random_vel = cfg.task.hit_random_vel
        # 添加随机化概率参数
        self.hit_random_vel_prob = cfg.task.get("hit_random_vel_prob", 0.3)  # 默认概率为0.3

        # 添加观测噪声参数
        self.obs_noise_std = cfg.task.get("obs_noise_std", 0.01)  # 观测噪声的标准差
        self.use_obs_noise = cfg.task.get("use_obs_noise", True)  # 是否使用观测噪声

        # 添加中心击球奖励参数
        self.use_center_hit_reward = cfg.task.get("use_center_hit_reward", True)  # 是否使用中心击球奖励
        self.center_hit_reward_weight = cfg.task.get("center_hit_reward_weight", 10.0)  # 中心击球奖励权重
        self.center_hit_reward_scale = cfg.task.get("center_hit_reward_scale", 5.0)  # 中心击球奖励衰减系数

        # 添加旋转奖励参数
        self.use_spin_reward = cfg.task.get("use_spin_reward", True)  # 是否使用旋转奖励
        self.reward_spin_weight = cfg.task.get("reward_spin_weight", 1.0)  # 旋转奖励权重

        # 添加击球距离惩罚参数
        self.use_hit_distance_penalty = cfg.task.get("use_hit_distance_penalty", False)  # 是否使用击球距离惩罚
        self.hit_distance_penalty_weight = cfg.task.get("hit_distance_penalty_weight", 0.5)  # 击球距离惩罚权重

        # 添加速度惩罚参数
        self.use_velocity_penalty = cfg.task.get("use_velocity_penalty", True)  # 是否使用速度惩罚
        self.velocity_penalty_weight = cfg.task.get("velocity_penalty_weight", 1.0)  # 速度惩罚权重

        # 添加偏航角惩罚参数
        self.use_yaw_penalty = cfg.task.get("use_yaw_penalty", False)  # 是否使用偏航角惩罚
        self.yaw_penalty_weight = cfg.task.get("yaw_penalty_weight", 10.0)  # 偏航角惩罚权重

        # 添加俯仰角和滚转角惩罚参数
        self.use_attitude_penalty = cfg.task.get("use_attitude_penalty", False)  # 是否使用姿态角惩罚
        self.attitude_penalty_weight = cfg.task.get("attitude_penalty_weight", 10.0)  # 姿态角惩罚权重
        self.max_attitude_angle = cfg.task.get("max_attitude_angle", 5.0) * torch.pi / 180.0  # 最大姿态角，默认5度

        # 添加无人机速度延迟步骤参数
        self.drone_vel_latent_step = cfg.task.get("drone_vel_latent_step", 0)  # 无人机速度延迟步骤，默认为0
        
        # 添加球体线性速度延迟步骤参数
        self.ball_linear_vel_latent_step = cfg.task.get("ball_linear_vel_latent_step", 2)  # 球体线性速度延迟步骤，默认为2

        # 添加位置惩罚参数
        self.use_pos_xy_penalty = cfg.task.get("use_pos_xy_penalty", True)  # 是否使用水平位置惩罚
        self.pos_xy_penalty_weight = cfg.task.get("pos_xy_penalty_weight", 1.0)  # 水平位置惩罚权重
        
        self.use_pos_z_penalty = cfg.task.get("use_pos_z_penalty", True)  # 是否使用垂直位置惩罚
        self.pos_z_penalty_weight = cfg.task.get("pos_z_penalty_weight", 0.2)  # 垂直位置惩罚权重

        # 添加距离裁剪参数
        self.use_distance_clip = cfg.task.get("use_distance_clip", True)  # 是否使用距离裁剪
        self.min_distance_clip = cfg.task.get("min_distance_clip", 0.2)  # 最小距离裁剪值，默认0.2米

        # 添加球高度奖励参数
        self.use_ball_height_reward = cfg.task.get("use_ball_height_reward", True)  # 是否使用球高度奖励
        self.target_ball_height = cfg.task.get("target_ball_height", 1.6)  # 目标球高度

        self.drone.initialize()

        self.ball = RigidPrimView(
            "/World/envs/env_*/ball",
            reset_xform_properties=False,
            track_contact_forces=False, # set false because we use contact sensor to get contact forces
            shape=(-1, 1)
        )
        self.ball.initialize()
        # self.ball_initial_z_vel = cfg.task.initial.get("ball_initial_z_vel", 0.0)
        self.racket_radius = cfg.task.get("racket_radius", 0.05)
        self.ball_initial_z_vel = 0.0


        self.init_ball_pos_dist = D.Uniform(
            torch.tensor([-0.07, -0.07, 1.5], device=self.device),
            torch.tensor([0.07, 0.07, 2.], device=self.device)
        )
        self.init_drone_pos_dist = D.Uniform(
            torch.tensor([-0.07, -0.07, 0.9], device=self.device),
            torch.tensor([0.07, 0.07, 1.1], device=self.device)
        )
        self.init_ball_vel_dist = D.Uniform(
            torch.tensor([0., 0., 0., 0.0, 0.0, 0.0], device=self.device),
            torch.tensor([0., 0., 0., 0.0, 0.0, 0.0], device=self.device)
            )
        self.init_drone_vel_dist = D.Uniform(
            torch.tensor([0., 0., 0., 0.0, 0.0, 0.0], device=self.device),
            torch.tensor([0., 0., 0., 0.0, 0.0, 0.0], device=self.device)
            )
        self.init_drone_rpy_dist = D.Uniform(
            torch.tensor([0., 0., 0.], device=self.device) / 180 * torch.pi,
            torch.tensor([0., 0., 0.], device=self.device) / 180 * torch.pi
        )


        if self.use_eval:
            self.init_ball_pos_dist = D.Uniform(
                torch.tensor([-0.07, -0.07, 1.5], device=self.device),
                torch.tensor([0.07, 0.07, 2.0], device=self.device)
            )
            self.init_drone_pos_dist = D.Uniform(
                torch.tensor([-0.07, -0.07, 0.9], device=self.device),
                torch.tensor([0.07, 0.07, 1.1], device=self.device)
            ) 
            self.init_ball_vel_dist = D.Uniform(
                torch.tensor([0., 0., 0., 0.0, 0.0, 0.0], device=self.device),
                torch.tensor([0., 0., 0., 0.0, 0.0, 0.0], device=self.device)
            )
            self.init_drone_vel_dist = D.Uniform(
                torch.tensor([0., 0., 0., 0.0, 0.0, 0.0], device=self.device),
                torch.tensor([0., 0., 0., 0.0, 0.0, 0.0], device=self.device)
            )
            self.init_drone_rpy_dist = D.Uniform(
                torch.tensor([0., 0., 0.], device=self.device) / 180 * torch.pi,
                torch.tensor([0., 0., 0.], device=self.device) / 180 * torch.pi
            )



        self.restitution_dist = D.Uniform(
            torch.tensor([self.restitution_range[0]], device=self.device),
            torch.tensor([self.restitution_range[1]], device=self.device)
        )
            
        self.hit_random_vel_dist = D.Uniform(
            torch.tensor([-0.2, -0.2, 0.0], device=self.device),
            torch.tensor([0.2, 0.2, 0.0], device=self.device)
        )
        self.last_hit_t = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.int64
        )
        
        # pos(3) + rpq(3) + vel(6) = 12
        # self.drone_initial_state = torch.zeros(self.num_envs, 12, device=self.device, dtype=torch.float)
        
        self.ball_traj_vis = []

        self.central_env_pos = Float3(
            *self.envs_positions[self.central_env_idx].tolist()
        )
        
        self.draw = _debug_draw.acquire_debug_draw_interface()

        self.last_linear_v = torch.zeros(self.num_envs, 1, device=self.device)
        self.last_angular_v = torch.zeros(self.num_envs, 1, device=self.device)
        self.last_linear_a = torch.zeros(self.num_envs, 1, device=self.device)
        self.last_angular_a = torch.zeros(self.num_envs, 1, device=self.device)
        self.last_linear_jerk = torch.zeros(self.num_envs, 1, device=self.device)
        self.last_angular_jerk = torch.zeros(self.num_envs, 1, device=self.device)
        # 添加last_hit_xy变量来存储上一次击球的xy位置
        self.last_hit_xy = torch.zeros(self.num_envs, 1, 2, device=self.device, dtype=torch.float32)


        self.prev_actions = torch.zeros(self.num_envs, self.num_drones, 4, device=self.device)


        self.ball_height_max = torch.zeros(self.num_envs, 1, device=self.device)
        self.racket_near_ball = torch.zeros((cfg.task.env.num_envs, 1), device=self.device, dtype=torch.bool)
        self.drone_near_ball = torch.zeros((cfg.task.env.num_envs, 1), device=self.device, dtype=torch.bool)
        self.ball_last_vel = torch.zeros(self.num_envs, 1, 3, device=self.device, dtype=torch.float32)
        self.ball_last_2_vel = torch.zeros(self.num_envs, 1, 3, device=self.device, dtype=torch.float32)
        self.linear_vel_last_2 = torch.zeros(self.num_envs, 1, 3, device=self.device, dtype=torch.float32)
        self.linear_vel_last = torch.zeros(self.num_envs, 1, 3, device=self.device, dtype=torch.float32)
        self.angular_vel_last_2 = torch.zeros(self.num_envs, 1, 3, device=self.device, dtype=torch.float32)
        self.angular_vel_last = torch.zeros(self.num_envs, 1, 3, device=self.device, dtype=torch.float32)
        self.last_hit_step = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.int64)
        self.hited_mark = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.bool)
        self.current_restitution = torch.ones(self.num_envs, 1, device=self.device, dtype=torch.float32) * self.base_restitution


    def _design_scene(self):
        drone_model = MultirotorBase.REGISTRY[self.cfg.task.drone_model]
        cfg = drone_model.cfg_cls(force_sensor=self.cfg.task.force_sensor)
        self.drone: MultirotorBase = drone_model(cfg=cfg)

        if not self.cfg.task.use_restitution_randomization:
            # if use restitution randomization, the material will be created in the isaac_env.py
            # so we don't need to create the material here
            material = materials.PhysicsMaterial(
                prim_path="/World/Physics_Materials/physics_material_0",
                restitution=self.base_restitution,
            )
            ball = objects.DynamicSphere(
                prim_path="/World/envs/env_0/ball",
                radius=self.ball_radius,
                mass=self.ball_mass,
                color=torch.tensor([1., .2, .2]),
                physics_material=material,
            )

        if self.use_local_usd:
            # use local usd resources
            usd_path = os.path.join(os.path.dirname(__file__), os.pardir, "assets", "default_environment.usd")
            kit_utils.create_ground_plane(
                "/World/defaultGroundPlane",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
                usd_path=usd_path
            )
        else:
            # use online usd resources
            kit_utils.create_ground_plane(
                "/World/defaultGroundPlane",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            )

        drone_prims = self.drone.spawn(translations=[(0.0, 0.0, 2.)])
        material_drone = materials.PhysicsMaterial(
            prim_path="/World/Physics_Materials/drone_material_0",
            restitution=self.base_restitution,
        )
        material_drone = UsdShade.Material(material_drone.prim)
        for drone_prim in drone_prims:
            collision_prim = drone_prim.GetPrimAtPath("base_link/collisions")
            binding_api = UsdShade.MaterialBindingAPI(collision_prim)
            binding_api.Bind(material_drone, UsdShade.Tokens.weakerThanDescendants, "physics")

        return ["/World/defaultGroundPlane"]

    def _set_specs(self):
        observation_dim = 24

        
        self.time_encoding_dim = 4
        if self.cfg.task.time_encoding:
            observation_dim += self.time_encoding_dim

        state_dim = observation_dim + self.time_encoding_dim
        self.observation_spec = (
            CompositeSpec(
                {
                    "agents": CompositeSpec(
                        {
                            "observation": UnboundedContinuousTensorSpec(
                                (1, observation_dim)
                            ),
                            "state": UnboundedContinuousTensorSpec(
                                (state_dim)
                            ),
                        }
                    )
                }
            )
            .expand(self.num_envs)
            .to(self.device)
        )
        self.action_spec = (
            CompositeSpec(
                {
                    "agents": CompositeSpec(
                        {
                            "action": torch.stack([self.drone.action_spec] * 1, dim=0),
                        }
                    )
                }
            )
            .expand(self.num_envs)
            .to(self.device)
        )
        self.reward_spec = (
            CompositeSpec(
                {
                    "agents": CompositeSpec(
                        {"reward": UnboundedContinuousTensorSpec((1, 1))}
                    )
                }
            )
            .expand(self.num_envs)
            .to(self.device)
        )
        self.done_spec = CompositeSpec({
            "done": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
            "terminated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
            "truncated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
            "wrong_hit": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
        }).expand(self.num_envs).to(self.device)
        self.agent_spec["drone"] = AgentSpec(
            "drone",
            1,
            observation_key=("agents", "observation"),
            state_key=("agents", "state"),
            action_key=("agents", "action"),
            reward_key=("agents", "reward"),
        )
        
        _stats_spec = CompositeSpec(
            {
                "return": UnboundedContinuousTensorSpec(1),
                "episode_len": UnboundedContinuousTensorSpec(1),
                "num_true_hits": UnboundedContinuousTensorSpec(1),
                "ball_vz": UnboundedContinuousTensorSpec(2),
                "drone_z_on_hit": UnboundedContinuousTensorSpec(2),
                "wrong_hit": UnboundedContinuousTensorSpec(1),
                "ball_too_low": UnboundedContinuousTensorSpec(1),
                "ball_too_high": UnboundedContinuousTensorSpec(1),
                "ball_too_far": UnboundedContinuousTensorSpec(1),
                "drone_too_low": UnboundedContinuousTensorSpec(1),
                "drone_too_high": UnboundedContinuousTensorSpec(1),
                "truncated": UnboundedContinuousTensorSpec(1),
                "done": UnboundedContinuousTensorSpec(1),
                "num_hits": UnboundedContinuousTensorSpec(1),

                "action_error_order1_mean": UnboundedContinuousTensorSpec(1),
                "action_error_order1_max": UnboundedContinuousTensorSpec(1),
                "smoothness_mean": UnboundedContinuousTensorSpec(1),
                "smoothness_max": UnboundedContinuousTensorSpec(1),
                
                "linear_v_max": UnboundedContinuousTensorSpec(1),
                "angular_v_max": UnboundedContinuousTensorSpec(1),
                "linear_a_max": UnboundedContinuousTensorSpec(1),
                "angular_a_max": UnboundedContinuousTensorSpec(1),
                "linear_jerk_max": UnboundedContinuousTensorSpec(1),
                "angular_jerk_max": UnboundedContinuousTensorSpec(1),
                "linear_v_mean": UnboundedContinuousTensorSpec(1),
                "angular_v_mean": UnboundedContinuousTensorSpec(1),
                "linear_a_mean": UnboundedContinuousTensorSpec(1),
                "angular_a_mean": UnboundedContinuousTensorSpec(1),
                "linear_jerk_mean": UnboundedContinuousTensorSpec(1),
                "angular_jerk_mean": UnboundedContinuousTensorSpec(1),
            }
        )
        self.stats_cfg: DictConfig = self.cfg.task.stats
        if self.stats_cfg.get("complete_reward_stats", False):
            _stats_spec.set("reward_score", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("reward_new_hit", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("reward_new_high", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("reward_rpos", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("penalty_pos_xy", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("penalty_pos_z", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("penalty_yaw", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("reward_action_smoothness", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("reward_ball_height", UnboundedContinuousTensorSpec(1))

        stats_spec = _stats_spec.expand(self.num_envs).to(self.device)

        info_spec = (
            CompositeSpec(
                {
                    "drone_state": UnboundedContinuousTensorSpec((self.drone.n, 13)),
                    "prev_action": torch.stack([self.drone.action_spec] * self.drone.n, 0).to(self.device),
                    "policy_action": torch.stack([self.drone.action_spec] * self.drone.n, 0).to(self.device),
                    # "prev_action": torch.stack([self.drone.action_spec] * self.drone.n, 0),
                    # "policy_action": torch.stack([self.drone.action_spec] * self.drone.n, 0),
                }
            )
            .expand(self.num_envs)
            .to(self.device)
        )
        
        self.observation_spec["stats"] = stats_spec
        self.observation_spec["info"] = info_spec

        self.stats = stats_spec.zero()
        self.info = info_spec.zero()

    def debug_draw_region(self, region=0):
        if region == 0:
            b_x = 1.5
            b_y = 1.5
            b_z_top = 1.0
            b_z_bot = 2.0
            height = b_z_top - b_z_bot
            color = [(0., 1., 0., 1.)]
        elif region == 1:
            b_x = 2.0
            b_y= 2.0
            b_z_top = 3.0
            b_z_bot = 0.0
            height = b_z_top - b_z_bot
            color = [(0.95, 0.43, 0.2, 1.)]
        # [topleft, topright, botleft, botright]
        

        points_start, points_end = rectangular_cuboid_edges(2*b_x, 2*b_y, b_z_bot, height)
        points_start = [_carb_float3_add(p, self.central_env_pos) for p in points_start]
        points_end = [_carb_float3_add(p, self.central_env_pos) for p in points_end]
        
        colors_line = color * len(points_start)
        sizes_line = [1.] * len(points_start)
        self.draw.draw_lines(points_start,
                                 points_end, colors_line, sizes_line)
    
    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids, self.training)

        if self.on_the_spot:
            drone_pos = torch.tensor([0., 0., 1.0], device=self.device)
            drone_rpy = torch.tensor([0., 0., 0.], device=self.device)
            drone_rot = euler_to_quaternion(drone_rpy)
        else:
            drone_pos = self.init_drone_pos_dist.sample((*env_ids.shape, 1))
            drone_rpy = self.init_drone_rpy_dist.sample((*env_ids.shape, 1))
            drone_rot = euler_to_quaternion(drone_rpy)
        
        self.drone.set_world_poses(
            drone_pos +
            self.envs_positions[env_ids].unsqueeze(1), drone_rot, env_ids
        )
        
        drone_vel = self.init_drone_vel_dist.sample((*env_ids.shape, 1))
        self.drone.set_velocities(drone_vel, env_ids)
        

        if self.cfg.task.use_restitution_randomization:
            restitution = self.restitution_dist.sample((*env_ids.shape, 1))
            for i, env_idx in enumerate(env_ids):
                material = materials.PhysicsMaterial(
                    prim_path=f"/World/Physics_Materials/physics_material_{env_idx}",
                    restitution=restitution[i].item()
                )
                self.current_restitution[env_idx] = restitution[i].item()
        # below is for check the ball restitution 
        # import omni.usd
        # from pxr import UsdPhysics, Sdf, Usd
        # stage = omni.usd.get_context().get_stage()

        # ball_path = "/World/envs/env_2/ball"
        # ball_prim = stage.GetPrimAtPath(ball_path)
        # if ball_prim:
        #     # 检查所有关系
        #     print(f"球体 {ball_path} 的所有关系:")
        #     for rel in ball_prim.GetRelationships():
        #         rel_name = rel.GetName()
        #         targets = rel.GetTargets()
        #         print(f"  关系: {rel_name}")
        #         print(f"  目标: {targets}")
        #         if  "physics" in rel_name.lower():
        #             for target_path in targets:
        #                 target_prim = stage.GetPrimAtPath(target_path)
        #                 if target_prim:
        #                     print(f"  可能的物理材质: {target_path}")
        #                     # 检查该prim的所有属性
        #                     for prop in target_prim.GetProperties():
        #                         if "restitution" in prop.GetName().lower():
        #                             print(f"  弹性系数属性: {prop.GetName()} = {prop.Get()}")

            

        if self.on_the_spot:
            ball_pos = torch.tensor([0., 0., 1.6], device=self.device)
        else:
            ball_pos = self.init_ball_pos_dist.sample((*env_ids.shape, 1))

        ball_rot = torch.tensor(
            [1., 0., 0., 0.], device=self.device).repeat(len(env_ids), 1)
        
        self.ball.set_world_poses(
            ball_pos +
            self.envs_positions[env_ids].unsqueeze(1), ball_rot, env_ids
        )

        ball_vel = self.init_ball_vel_dist.sample((*env_ids.shape, 1))
        self.ball.set_velocities(ball_vel, env_ids)
        self.ball.set_masses(
            torch.ones_like(env_ids,dtype=torch.float)*self.ball_mass, env_ids)
        
        self.stats[env_ids] = 0.
        self.last_hit_t[env_ids] = - 999
        self.last_hit_step[env_ids] = -100
        self.hited_mark[env_ids] = False


        # set last values
        self.last_linear_v[env_ids] = torch.zeros_like(self.last_linear_v[env_ids])
        self.last_angular_v[env_ids] = torch.zeros_like(self.last_angular_v[env_ids])
        self.last_linear_a[env_ids] = torch.zeros_like(self.last_linear_a[env_ids])
        self.last_angular_a[env_ids] = torch.zeros_like(self.last_angular_a[env_ids])
        self.last_linear_jerk[env_ids] = torch.zeros_like(self.last_linear_jerk[env_ids])
        self.last_angular_jerk[env_ids] = torch.zeros_like(self.last_angular_jerk[env_ids])
        self.last_hit_xy[env_ids] = torch.zeros_like(self.last_hit_xy[env_ids])

        # CTBR
        cmd_init = 2.0 * (self.drone.throttle[env_ids]) ** 2 - 1.0
        self.info['prev_action'][env_ids, :, 3] = cmd_init.mean(dim=-1)
        self.prev_actions[env_ids] = self.info['prev_action'][env_ids].clone()
        
        # Draw
        if (env_ids == self.central_env_idx).any() and self._should_render(0):
            self.ball_traj_vis.clear()
            self.draw.clear_lines()
            self.debug_draw_region(0)
            self.debug_draw_region(1)
        
        ball_initial_vel = torch.zeros(len(env_ids), 6, device=self.device)
        ball_initial_vel[:, 2] = self.ball_initial_z_vel
        self.ball.set_velocities(ball_initial_vel, env_ids)
        self.ball_last_vel[env_ids] = ball_vel[..., :3]
        self.ball_last_2_vel[env_ids] = ball_vel[..., :3]
        self.linear_vel_last_2[env_ids] = drone_vel[..., :3]
        self.linear_vel_last[env_ids] = drone_vel[..., :3]
        self.angular_vel_last_2[env_ids] = drone_vel[..., 3:]
        self.angular_vel_last[env_ids] = drone_vel[..., 3:]
        self.racket_near_ball[env_ids] = False
        self.drone_near_ball[env_ids] = False
        self.ball_height_max[env_ids] = 0.


    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("agents", "action")]

        # CTBR
        self.info["prev_action"] = tensordict[("info", "prev_action")]
        self.prev_actions = self.info["prev_action"].clone()

        self.info["policy_action"] = tensordict[("info", "policy_action")]
        self.policy_actions = tensordict[("info", "policy_action")].clone()

        self.action_error_order1 = tensordict[("stats", "action_error_order1")].clone()
        self.stats["action_error_order1_mean"].add_(self.action_error_order1.mean(dim=-1).unsqueeze(-1))
        self.stats["action_error_order1_max"].set_(torch.max(self.stats["action_error_order1_max"], self.action_error_order1.mean(dim=-1).unsqueeze(-1)))

        self.effort = self.drone.apply_action(actions)

    def _post_sim_step(self, tensordict: TensorDictBase):
        pass

    def _compute_state_and_obs(self):
        self.root_state = self.drone.get_state()
        self.info["drone_state"][:] = self.root_state[..., :13]
        
        self.ball_pos, ball_rot = self.get_env_poses(self.ball.get_world_poses())
        self.ball_linear_vel = self.ball.get_velocities()[..., :3]
        root_state = self.root_state.clone()
        pos = root_state[..., :3]

        
        
        # relative position and heading
        self.rpos = self.ball_pos - self.drone.pos # (E,1,3)

        self.rpos_ball = self.drone.pos - self.ball_pos # (E,1,3)


        pos, rot, linear_vel, angular_vel, linear_vel_b, angular_vel_b, heading, lateral, up, throttle = torch.split(
            self.root_state, split_size_or_sections=[3, 4, 3, 3, 3, 3, 3, 3, 3, 4], dim=-1
        )
        rot = root_state[..., 3:7]
        rot[..., :] = torch.where((rot[..., 0] < 0).unsqueeze(-1), -rot[..., :], rot[..., :]) # make sure w of (w,x,y,z) is positive
        self.drone_rot = rot
        
        rpy = quaternion_to_euler(rot)
        self.drone_pos = pos.squeeze(1)
        self.drone_rpy = rpy.squeeze(1)
        self.drone_vel = linear_vel.squeeze(1)
        self.drone_angular_vel = angular_vel.squeeze(1)
        self.roll = rpy[..., 0]
        self.pitch = rpy[..., 1]
        self.yaw = rpy[..., 2]
        
        self.info["roll"] = self.roll
        self.info["pitch"] = self.pitch
        self.info["yaw"] = self.yaw

        if self.drone_vel_latent_step ==0:
            linear_vel = self.drone.vel[..., :3]
            angular_vel = self.drone.vel[..., 3:]
        elif self.drone_vel_latent_step ==1:
            linear_vel = self.linear_vel_last
            angular_vel = self.angular_vel_last
        elif self.drone_vel_latent_step ==2:
            linear_vel = self.linear_vel_last_2
            angular_vel = self.angular_vel_last_2
        else:
            raise ValueError(f"drone_vel_latent_step must be 0, 1, or 2, but got {self.drone_vel_latent_step}")

        obs = [
            pos, 
            linear_vel,
            heading,
            lateral,
            up,
            self.rpos,
            self.ball_pos,
        ]

        if self.ball_linear_vel_latent_step ==0:
            obs.append(self.ball_linear_vel)
        elif self.ball_linear_vel_latent_step ==1:
            obs.append(self.ball_last_vel)
        elif self.ball_linear_vel_latent_step ==2:
            obs.append(self.ball_last_2_vel)
        else:
            raise ValueError(f"ball_linear_vel_latent_step must be 0, 1, or 2, but got {self.ball_linear_vel_latent_step}")

        
        if self.time_encoding:
            t = (self.progress_buf / self.max_episode_length).unsqueeze(-1)
            obs.append(t.expand(-1, self.time_encoding_dim).unsqueeze(1)) # obs_dim: root_state + rpos(3) + ball_vel(3) + [restitution(1) if enabled] + time_encoding(4)
        obs = torch.cat(obs, dim=-1)
        
        self.linear_vel_last_2 = self.linear_vel_last.clone()
        self.linear_vel_last = self.drone.vel[..., :3].clone()
        self.angular_vel_last_2 = self.angular_vel_last.clone()
        self.angular_vel_last = self.drone.vel[..., 3:].clone()



        if self.hit_random_vel and self.hited_mark.any():
            # 生成随机数来决定是否添加随机化
            random_trigger = torch.rand(len(self.hited_mark), device=self.device) < self.hit_random_vel_prob
            if random_trigger.any():

                random_vel = self.hit_random_vel_dist.sample((random_trigger.sum().item(), 1))
                ball_vel = self.ball.get_velocities()
                ball_vel[random_trigger, :, :3] += random_vel
                self.ball.set_velocities(ball_vel)
            self.hited_mark[:] = False  # 重置标记


        # add time encoding
        t = (self.progress_buf / self.max_episode_length).unsqueeze(-1)
        state = torch.concat([obs, t.expand(-1, self.time_encoding_dim).unsqueeze(1)], dim=-1).squeeze(1)

        self.stats["smoothness_mean"].add_(self.drone.throttle_difference)
        self.stats["smoothness_max"].set_(torch.max(self.drone.throttle_difference, self.stats["smoothness_max"]))
        # linear_v, angular_v
        self.linear_v = torch.norm(self.root_state[..., 7:10], dim=-1)
        self.angular_v = torch.norm(self.root_state[..., 10:13], dim=-1)
        self.stats["linear_v_max"].set_(torch.max(self.stats["linear_v_max"], torch.abs(self.linear_v)))
        self.stats["linear_v_mean"].add_(self.linear_v)
        self.stats["angular_v_max"].set_(torch.max(self.stats["angular_v_max"], torch.abs(self.angular_v)))
        self.stats["angular_v_mean"].add_(self.angular_v)
        # linear_a, angular_a
        self.linear_a = torch.abs(self.linear_v - self.last_linear_v) / self.dt
        self.angular_a = torch.abs(self.angular_v - self.last_angular_v) / self.dt
        self.stats["linear_a_max"].set_(torch.max(self.stats["linear_a_max"], torch.abs(self.linear_a)))
        self.stats["linear_a_mean"].add_(self.linear_a)
        self.stats["angular_a_max"].set_(torch.max(self.stats["angular_a_max"], torch.abs(self.angular_a)))
        self.stats["angular_a_mean"].add_(self.angular_a)
        # linear_jerk, angular_jerk
        self.linear_jerk = torch.abs(self.linear_a - self.last_linear_a) / self.dt
        self.angular_jerk = torch.abs(self.angular_a - self.last_angular_a) / self.dt
        self.stats["linear_jerk_max"].set_(torch.max(self.stats["linear_jerk_max"], torch.abs(self.linear_jerk)))
        self.stats["linear_jerk_mean"].add_(self.linear_jerk)
        self.stats["angular_jerk_max"].set_(torch.max(self.stats["angular_jerk_max"], torch.abs(self.angular_jerk)))
        self.stats["angular_jerk_mean"].add_(self.angular_jerk)

        # set last
        self.last_linear_v = self.linear_v.clone()
        self.last_angular_v = self.angular_v.clone()
        self.last_linear_a = self.linear_a.clone()
        self.last_angular_a = self.angular_a.clone()
        self.last_linear_jerk = self.linear_jerk.clone()
        self.last_angular_jerk = self.angular_jerk.clone()
        
        if self._should_render(0):
            central_env_pos = self.envs_positions[self.central_env_idx]
            ball_plot_pos = (
                self.ball_pos[self.central_env_idx]+central_env_pos).tolist()  # [2, 3]
            if len(self.ball_traj_vis) > 1:
                point_list_0 = self.ball_traj_vis[-1]
                point_list_1 = ball_plot_pos
                colors = [(.1, 1., .1, 1.)]
                sizes = [1.5]
                self.draw.draw_lines(point_list_0, point_list_1, colors, sizes)

            self.ball_traj_vis.append(ball_plot_pos)

        return TensorDict(
            {
                "agents": {
                    "observation": obs,
                    "state":state,
                },
                "stats": self.stats,
                "info": self.info,
            },
            self.num_envs,
        )
    def check_ball_near_racket(self, racket_radius, cylinder_height_coeff):
        z_direction_local = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        z_direction_world = quat_rotate(self.drone_rot, z_direction_local) # (E,N,3)
        normal_vector_world = z_direction_world / torch.norm(z_direction_world, dim=-1).unsqueeze(-1)  # (E,N,3)
        cylinder_bottom_center = self.drone.pos + normal_vector_world * 0.055  # (E,N,3) cylinder bottom center, 0.055 is the distance between racket and drone center
        cylinder_axis = cylinder_height_coeff * self.ball_radius * normal_vector_world 

        ball_to_bottom = self.ball_pos - cylinder_bottom_center  # (E,N,3)
        projection_ratio = torch.sum(ball_to_bottom * cylinder_axis, dim=-1) / torch.sum(cylinder_axis * cylinder_axis, dim=-1)  # (E,N) projection of ball_to_bottom on cylinder_axis / cylinder_axis
        within_height = (projection_ratio >= 0) & (projection_ratio <= 1)  # (E,N)

        projection_point = cylinder_bottom_center + projection_ratio.unsqueeze(-1) * cylinder_axis  # (E,N,3)
        distance_to_axis = torch.norm(self.ball_pos - projection_point, dim=-1)  # (E,N)
        within_radius = distance_to_axis <= racket_radius  # (E,N)

        return (within_height & within_radius)  # (E,N)
    
    def check_hit(self, sim_dt, racket_radius=0.2, cylinder_height_coeff=2.0):
        racket_near_ball_last_step = self.racket_near_ball.clone()
        drone_near_ball_last_step = self.drone_near_ball.clone()

        self.racket_near_ball = self.check_ball_near_racket(racket_radius=racket_radius, cylinder_height_coeff=cylinder_height_coeff)  # (E,N)
        self.drone_near_ball = (torch.norm(self.rpos_ball, dim=-1) < 0.2) # (E,N)

        ball_vel_z_change = ((self.ball_linear_vel[..., 2] - self.ball_last_vel[..., 2]) > 9.8 * sim_dt) # (E,1)
        ball_vel_x_y_change = (self.ball_linear_vel[..., :2] - self.ball_last_vel[..., :2]).norm(dim=-1) > 0.5 # (E,1)
        ball_vel_change = ball_vel_z_change | ball_vel_x_y_change # (E,1)
        
        drone_hit_ball = (drone_near_ball_last_step | self.drone_near_ball) & ball_vel_change # (E,N)
        racket_hit_ball = (racket_near_ball_last_step | self.racket_near_ball) & ball_vel_change # (E,N)
        racket_hit_ball = racket_hit_ball & (self.progress_buf.unsqueeze(-1) - self.last_hit_step > 3) # (E,N)
        drone_hit_ball = drone_hit_ball & (self.progress_buf.unsqueeze(-1) - self.last_hit_step > 3) # (E,N)
        return racket_hit_ball, drone_hit_ball

    def _compute_reward_and_done(self):

        racket_hit_ball, drone_hit_ball = self.check_hit(sim_dt=self.dt,racket_radius=self.racket_radius)
        
        hit = racket_hit_ball # (E, 1)
        true_hit_step_gap = self.true_hit_step_gap # 35 * 0.02s = 0.7s = 0.6 meter
        true_hit = hit & (((self.progress_buf - self.last_hit_t) > true_hit_step_gap).unsqueeze(-1)) # (E,1)

        wrong_hit_step_gap = 15 # 10 * 0.02s = 0.2s (hit the ball too frequently)
        wrong_hit = hit & (((self.progress_buf - self.last_hit_t) <= wrong_hit_step_gap).unsqueeze(-1)) # (E,1) # TODO: seems it doesn't work
        self.last_hit_t = hit.squeeze(-1).long() * self.progress_buf.long() + (1 - hit.squeeze(-1).long()) * self.last_hit_t

        reward_new_high = 50.0 * (
            (self.ball_last_vel[..., 2] >= 0) & 
            (self.ball_linear_vel[..., 2] < 0) & 
            (self.ball_pos[..., 2] > 1.6) & 
            (self.ball_pos[..., 2] < 1.85)
        ).float()

        # 设置碰撞标记
        self.hited_mark = hit.clone()

        self.ball_last_2_vel = self.ball_last_vel.clone()
        self.ball_last_vel = self.ball_linear_vel.clone()
        self.last_hit_step[racket_hit_ball] = self.progress_buf[racket_hit_ball.any(-1)].long()

        # 计算球拍中心与球的距离
        z_direction_local = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        z_direction_world = quat_rotate(self.drone_rot, z_direction_local) # (E,N,3)
        normal_vector_world = z_direction_world / torch.norm(z_direction_world, dim=-1).unsqueeze(-1)  # (E,N,3)
        cylinder_bottom_center = self.drone.pos + normal_vector_world * 0.055  # (E,N,3) cylinder bottom center, 0.055 is the distance between racket and drone center
        cylinder_axis = 2.0 * self.ball_radius * normal_vector_world 
        ball_to_bottom = self.ball_pos - cylinder_bottom_center  # (E,N,3)
        projection_ratio = torch.sum(ball_to_bottom * cylinder_axis, dim=-1) / torch.sum(cylinder_axis * cylinder_axis, dim=-1)  # (E,N)
        projection_point = cylinder_bottom_center + projection_ratio.unsqueeze(-1) * cylinder_axis  # (E,N,3)
        distance_to_axis = torch.norm(self.ball_pos - projection_point, dim=-1)  # (E,N)
        
        # spin reward, fixed z
        spin = torch.square(self.drone.vel_b[..., -1])
        reward_spin = torch.zeros_like(spin)
        if self.use_spin_reward:
            reward_spin = self.reward_spin_weight * 0.5 / (1.0 + torch.square(spin))

        # 添加对击球时姿态角度的惩罚
        pitch_penalty = torch.zeros_like(self.pitch)
        roll_penalty = torch.zeros_like(self.roll)
        if self.use_attitude_penalty:
            pitch_penalty = torch.where(
                true_hit,
                self.attitude_penalty_weight * torch.relu(self.pitch.abs() - self.max_attitude_angle),  # 只在超过设定角度时惩罚
                torch.zeros_like(self.pitch)
            )
            roll_penalty = torch.where(
                true_hit,
                self.attitude_penalty_weight * torch.relu(self.roll.abs() - self.max_attitude_angle),  # 只在超过设定角度时惩罚
                torch.zeros_like(self.roll)
            )

        delta_z = (self.drone.pos[..., 2] - 1.0).abs() # (E,1)
        delta_xy = (self.drone.pos[..., :2]).norm(dim=-1) # (E,1)
        delta_v_xy = (self.ball_linear_vel[..., :2]).norm(dim=-1) # (E,1)


        reward_ball_height = torch.ones_like(self.ball_height_max)
        if self.use_ball_height_reward:
            #reward for ball's height
            self.ball_height_max = torch.where(
                self.ball_pos[..., 2] > self.ball_height_max,
                self.ball_pos[..., 2],
                self.ball_height_max
                )
            reward_ball_height = torch.exp(-(self.ball_height_max - self.target_ball_height)**2)
            self.ball_height_max[true_hit.squeeze(-1)] = 0.0
        
        reward_score = 100 * 1 / (1 + delta_z) * true_hit.float() * reward_ball_height
        reward_new_hit = 50.0 * (
            (self.drone.pos[..., 2].squeeze(-1) > 0.85) &
            (self.drone.pos[..., 2].squeeze(-1) < 1.15) &
            true_hit.squeeze(-1)
        ).float().unsqueeze(-1)
        

        # 计算距离，根据开关决定是否裁剪
        distance = torch.norm(self.rpos[..., :2], dim=-1)
        if self.use_distance_clip:
            # 将距离限制在最小值，这样当距离小于最小值时会得到最大奖励
            clipped_distance = torch.clamp(distance, min=self.min_distance_clip)
        else:
            clipped_distance = distance
            
        reward_rpos = 1 / (1 + clipped_distance)
        
        # 水平位置惩罚
        penalty_pos_xy = torch.zeros_like(delta_xy)
        if self.use_pos_xy_penalty:
            penalty_pos_xy = self.pos_xy_penalty_weight * delta_xy
        
        # 速度惩罚
        penalty_v_xy = torch.zeros_like(delta_v_xy)
        if self.use_velocity_penalty:
            penalty_v_xy = self.velocity_penalty_weight * delta_v_xy
        
        # 垂直位置惩罚
        penalty_pos_z = torch.zeros_like(delta_z)
        if self.use_pos_z_penalty:
            penalty_pos_z = self.pos_z_penalty_weight * delta_z
        
        # 偏航角惩罚
        penalty_yaw = torch.zeros_like(self.yaw)
        if self.use_yaw_penalty:
            penalty_yaw = self.yaw_penalty_weight * self.yaw.abs()
        
        # 计算击球位置奖励
        reward_hit_pos = torch.zeros_like(distance_to_axis)
        if self.use_center_hit_reward:
            reward_hit_pos = self.center_hit_reward_weight * torch.exp(-self.center_hit_reward_scale * distance_to_axis) * true_hit.float()
        
        # 计算连续两次击球位置的距离惩罚
        hit_distance = torch.norm(self.ball_pos[..., :2] - self.last_hit_xy, dim=-1)  # 计算当前击球位置与上次击球位置的距离
        hit_distance_penalty = torch.zeros_like(hit_distance)
        if self.use_hit_distance_penalty:
            hit_distance_penalty = self.hit_distance_penalty_weight * hit_distance * true_hit.float()  # 只在true hit时计算惩罚

        # 更新last_hit_xy
        self.last_hit_xy = torch.where(true_hit.unsqueeze(-1), self.ball_pos[..., :2], self.last_hit_xy)

        not_begin_flag = (self.progress_buf > 1).unsqueeze(1)
        reward_action_smoothness = self.reward_action_smoothness_weight * torch.exp(-self.action_error_order1) * not_begin_flag.float()
        reward = (
            reward_new_hit
            + reward_new_high
            # reward_score
            + reward_score * reward_spin 
            + reward_rpos
            + reward_hit_pos
            - hit_distance_penalty
            - penalty_pos_xy 
            - penalty_v_xy
            # - penalty_pos_z 
            - penalty_yaw
            - pitch_penalty
            - roll_penalty
            + reward_action_smoothness
            # + reward_ball_height
        )

        terminated = (
            (self.drone.pos[..., 2] < 0.3) | (self.drone.pos[..., 2] > 3.5) # z direction
            | (self.ball_pos[..., 2] < 0.2) | (self.ball_pos[..., 2] > 4.) # z direction
            | (self.ball_pos[..., :2].abs() > 2.).any(-1) # x,y direction
            | wrong_hit # hit the ball too frequently
        )
        truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
        done = truncated | terminated

        ep_len = self.progress_buf.unsqueeze(-1)

        # episode结束的stats才会上传到wandb（在_reset()函数里实现）
        self.stats["return"].add_(reward)
        self.stats["episode_len"][:] = ep_len
        
        self.stats["ball_vz"][true_hit.squeeze(-1), 0] += self.ball_linear_vel[true_hit.squeeze(-1), 0, 2] # ball velocity in z direction when true_hit
        self.stats["ball_vz"][true_hit.squeeze(-1), 1] += 1 # true_hit times
        
        self.stats["drone_z_on_hit"][true_hit.squeeze(-1), 0] += self.drone_pos[true_hit.squeeze(-1), 2] # drone z position when true_hit
        self.stats["drone_z_on_hit"][true_hit.squeeze(-1), 1] += 1 # true_hit times

        self.stats["done"].add_(done.float())
        self.stats["wrong_hit"].add_(wrong_hit.float())
        self.stats["ball_too_low"].add_(self.ball_pos[..., 2] < 0.2).float()
        self.stats["ball_too_high"].add_(self.ball_pos[..., 2] > 4.).float()
        self.stats["ball_too_far"].add_((self.ball_pos[..., :2].abs() > 2.).any(-1).float())
        self.stats["drone_too_low"].add_(self.drone.pos[..., 2] < 0.3).float()
        self.stats["drone_too_high"].add_(self.drone.pos[..., 2] > 3.5).float()
        self.stats["truncated"].add_(truncated.float())
        self.stats["num_true_hits"].add_(true_hit.float())
        self.stats["num_hits"].add_(hit.float())

        self.stats['action_error_order1_mean'].div_(
            torch.where(done, ep_len, torch.ones_like(ep_len))
        )
        self.stats['smoothness_mean'].div_(
            torch.where(done, ep_len, torch.ones_like(ep_len))
        )
        self.stats["linear_v_mean"].div_(
            torch.where(done, ep_len, torch.ones_like(ep_len))
        )
        self.stats["angular_v_mean"].div_(
            torch.where(done, ep_len, torch.ones_like(ep_len))
        )
        self.stats["linear_a_mean"].div_(
            torch.where(done, ep_len, torch.ones_like(ep_len))
        )
        self.stats["angular_a_mean"].div_(
            torch.where(done, ep_len, torch.ones_like(ep_len))
        )
        self.stats["linear_jerk_mean"].div_(
            torch.where(done, ep_len, torch.ones_like(ep_len))
        )
        self.stats["angular_jerk_mean"].div_( 
            torch.where(done, ep_len, torch.ones_like(ep_len))
        )

        if self.stats_cfg.get("complete_reward_stats", False):
            self.stats["reward_score"].add_(reward_score)
            self.stats["reward_new_hit"].add_(reward_new_hit)
            self.stats["reward_new_high"].add_(reward_new_high)
            self.stats["reward_rpos"].add_(reward_rpos)
            self.stats["penalty_pos_xy"].add_(penalty_pos_xy)
            self.stats["penalty_pos_z"].add_(penalty_pos_z)
            self.stats["penalty_yaw"].add_(penalty_yaw)
            self.stats["reward_action_smoothness"].add_(reward_action_smoothness)
            self.stats["reward_ball_height"].add_(reward_ball_height)
            self.stats["reward_hit_pos"] = reward_hit_pos
            self.stats["hit_distance_penalty"] = hit_distance_penalty
            
            if self.use_attitude_penalty:
                self.stats["penalty_pitch"] = pitch_penalty
                self.stats["penalty_roll"] = roll_penalty
            
            if self.use_velocity_penalty:
                self.stats["penalty_v_xy"] = penalty_v_xy
            
            if self.use_spin_reward:
                self.stats["reward_spin"] = reward_spin

        return TensorDict(
            {
                "agents": {
                    "reward": reward.unsqueeze(-1)
                },
                "done": done,
                "terminated": terminated,
                "truncated": truncated
            },
            self.num_envs,
        )
