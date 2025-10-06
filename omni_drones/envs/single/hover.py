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


import functorch
import torch
import torch.distributions as D

import omni.isaac.core.utils.prims as prim_utils

from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.robots.drone import MultirotorBase
from omni_drones.views import ArticulationView, RigidPrimView
from omni_drones.utils.torch import euler_to_quaternion, quat_axis

from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec, DiscreteTensorSpec
import os

def attach_payload(parent_path):
    from omni.isaac.core import objects
    import omni.physx.scripts.utils as script_utils
    from pxr import UsdPhysics

    payload_prim = objects.DynamicCuboid(
        prim_path=parent_path + "/payload",
        scale=torch.tensor([0.1, 0.1, .15]),
        mass=0.0001
    ).prim

    parent_prim = prim_utils.get_prim_at_path(parent_path + "/base_link")
    stage = prim_utils.get_current_stage()
    joint = script_utils.createJoint(stage, "Prismatic", payload_prim, parent_prim)
    UsdPhysics.DriveAPI.Apply(joint, "linear")
    joint.GetAttribute("physics:lowerLimit").Set(-0.15)
    joint.GetAttribute("physics:upperLimit").Set(0.15)
    joint.GetAttribute("physics:axis").Set("Z")
    joint.GetAttribute("drive:linear:physics:damping").Set(10.)
    joint.GetAttribute("drive:linear:physics:stiffness").Set(10000.)


class Hover(IsaacEnv):
    r"""
    A basic control task. The goal for the agent is to maintain a stable
    position and heading in mid-air without drifting. This task is designed
    to serve as a sanity check.

    ## Observation
    The observation space consists of the following part:

    - `rpos` (3): The position relative to the target hovering position.
    - `root_state` (16 + `num_rotors`): The basic information of the drone (except its position), 
      containing its rotation (in quaternion), velocities (linear and angular), 
      heading and up vectors, and the current throttle.
    - `rheading` (3): The difference between the reference heading and the current heading.
    - `time_encoding` (optional): The time encoding, which is a 4-dimensional vector encoding the current
      progress of the episode.

    ## Reward
    - `pos`: Reward computed from the position error to the target position.
    - `heading_alignment`: Reward computed from the alignment of the heading to the target heading.
    - `up`: Reward computed from the uprightness of the drone to discourage large tilting.
    - `spin`: Reward computed from the spin of the drone to discourage spinning.
    - `effort`: Reward computed from the effort of the drone to optimize the
      energy consumption.
    - `action_smoothness`: Reward that encourages smoother drone actions, computed based on the throttle difference of the drone.

    The total reward is computed as follows:

    ```{math}
        r = r_\text{pos} + r_\text{pos} * (r_\text{up} + r_\text{spin}) + r_\text{effort} + r_\text{action_smoothness}
    ```
        
    ## Episode End
    The episode ends when the drone mishebaves, i.e., it crashes into the ground or flies too far away:

    ```{math}
        d_\text{pos} > 4 \text{ or } x^w_z < 0.2
    ```
    
    or when the episode reaches the maximum length.


    ## Config

    | Parameter               | Type  | Default   | Description |
    |-------------------------|-------|-----------|-------------|
    | `drone_model`           | str   | "firefly" | Specifies the model of the drone being used in the environment. |
    | `reward_distance_scale` | float | 1.2       | Scales the reward based on the distance between the drone and its target. |
    | `time_encoding`         | bool  | True      | Indicates whether to include time encoding in the observation space. If set to True, a 4-dimensional vector encoding the current progress of the episode is included in the observation. If set to False, this feature is not included. |
    | `has_payload`           | bool  | False     | Indicates whether the drone has a payload attached. If set to True, it means that a payload is attached; otherwise, if set to False, no payload is attached. |


    """
    def __init__(self, cfg, headless):
        # self.reward_effort_weight = cfg.task.reward_effort_weight
        self.reward_action_smoothness_weight = cfg.task.reward_action_smoothness_weight
        self.reward_distance_scale = cfg.task.reward_distance_scale
        self.reward_up_weight = cfg.task.reward_up_weight
        self.reward_spin_weight = cfg.task.reward_spin_weight
        self.time_encoding = cfg.task.time_encoding
        self.randomization = cfg.task.get("randomization", {})
        self.has_payload = "payload" in self.randomization.keys()
        self.use_ctbr = True if cfg.task.action_transform=="PIDrate_FM" else False

        super().__init__(cfg, headless)

        self.drone.initialize()
        if "drone" in self.randomization:
            self.drone.setup_randomization(self.randomization["drone"])
        if "payload" in self.randomization:
            payload_cfg = self.randomization["payload"]
            self.payload_z_dist = D.Uniform(
                torch.tensor([payload_cfg["z"][0]], device=self.device),
                torch.tensor([payload_cfg["z"][1]], device=self.device)
            )
            self.payload_mass_dist = D.Uniform(
                torch.tensor([payload_cfg["mass"][0]], device=self.device),
                torch.tensor([payload_cfg["mass"][1]], device=self.device)
            )
            self.payload = RigidPrimView(
                f"/World/envs/env_*/{self.drone.name}_*/payload",
                reset_xform_properties=False,
                shape=(-1, self.drone.n)
            )
            self.payload.initialize()
        
        self.target_vis = ArticulationView(
            "/World/envs/env_*/target",
            reset_xform_properties=False
        )
        self.target_vis.initialize()
        self.init_poses = self.drone.get_world_poses(clone=True)
        self.init_vels = torch.zeros_like(self.drone.get_velocities())

        self.init_pos_dist = D.Uniform(
            torch.tensor([-1., -1., .5], device=self.device),
            torch.tensor([1., 1., 1.2], device=self.device)
        )
        self.init_rpy_dist = D.Uniform(
            torch.tensor([-0.1, -0.1, 0.], device=self.device) * torch.pi,
            torch.tensor([0.1, 0.1, 0.5], device=self.device) * torch.pi
        )
        self.target_rpy_dist = D.Uniform(
            torch.tensor([0., 0., 0.], device=self.device) * torch.pi,
            torch.tensor([0., 0., 0.], device=self.device) * torch.pi
        )

        self.target_pos = torch.tensor([[0.0, 0.0, 1.6]], device=self.device)
        self.alpha = 0.8

        if self.use_ctbr:
            self.last_linear_v = torch.zeros(self.num_envs, 1, device=self.device)
            self.last_angular_v = torch.zeros(self.num_envs, 1, device=self.device)
            self.last_linear_a = torch.zeros(self.num_envs, 1, device=self.device)
            self.last_angular_a = torch.zeros(self.num_envs, 1, device=self.device)

            self.prev_actions = torch.zeros(self.num_envs, 1, 4, device=self.device)

    def _design_scene(self):
        import omni_drones.utils.kit as kit_utils
        import omni.isaac.core.utils.prims as prim_utils

        drone_model = MultirotorBase.REGISTRY[self.cfg.task.drone_model]
        cfg = drone_model.cfg_cls(force_sensor=self.cfg.task.force_sensor)
        self.drone: MultirotorBase = drone_model(cfg=cfg)
        drone_prim = self.drone.spawn(translations=[(0.0, 0.0, 0.05)])[0]

        target_vis_prim = prim_utils.create_prim(
            prim_path="/World/envs/env_0/target",
            usd_path=self.drone.usd_path,
            translation=(0.0, 0.0, 1.7),
        )

        kit_utils.set_nested_collision_properties(
            target_vis_prim.GetPath(), 
            collision_enabled=False
        )
        kit_utils.set_nested_rigid_body_properties(
            target_vis_prim.GetPath(),
            disable_gravity=True
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
        
        if self.has_payload:
            attach_payload(drone_prim.GetPath().pathString)
        return ["/World/defaultGroundPlane"]

    def _set_specs(self):
        # drone_state_dim = self.drone.state_spec.shape[-1]
        observation_dim = 15
        self.time_encoding_dim = 4
        if self.cfg.task.time_encoding:
            observation_dim += self.time_encoding_dim

        state_dim = observation_dim + 4

        self.observation_spec = CompositeSpec({
            "agents": CompositeSpec({
                "observation": UnboundedContinuousTensorSpec((1, observation_dim), device=self.device),
                "state": UnboundedContinuousTensorSpec((state_dim), device=self.device), # add motor speed
            })
        }).expand(self.num_envs).to(self.device)
        self.action_spec = CompositeSpec({
            "agents": CompositeSpec({
                "action": self.drone.action_spec.unsqueeze(0),
            })
        }).expand(self.num_envs).to(self.device)
        self.reward_spec = CompositeSpec({
            "agents": CompositeSpec({
                "reward": UnboundedContinuousTensorSpec((1, 1))
            })
        }).expand(self.num_envs).to(self.device)
        self.done_spec = CompositeSpec({
            "done": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
            "terminated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
            "truncated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
        }).expand(self.num_envs).to(self.device)

        self.agent_spec["drone"] = AgentSpec(
            "drone", 1,
            observation_key=("agents", "observation"),
            action_key=("agents", "action"),
            reward_key=("agents", "reward"),
            state_key=("agents", "state")
        )

        if self.use_ctbr:
            stats_spec = CompositeSpec({
                "return": UnboundedContinuousTensorSpec(1),
                "pos_bonus": UnboundedContinuousTensorSpec(1),
                "reward_pos": UnboundedContinuousTensorSpec(1),
                "reward_up": UnboundedContinuousTensorSpec(1),
                "reward_spin": UnboundedContinuousTensorSpec(1),
                "reward_action_smoothness": UnboundedContinuousTensorSpec(1),
                "episode_len": UnboundedContinuousTensorSpec(1),
                "hover_error": UnboundedContinuousTensorSpec(1),
                "hover_error_ema": UnboundedContinuousTensorSpec(1),
                "action_error_order1_mean": UnboundedContinuousTensorSpec(1),
                "action_error_order1_max": UnboundedContinuousTensorSpec(1),
                "smoothness_mean": UnboundedContinuousTensorSpec(1),
                "smoothness_max": UnboundedContinuousTensorSpec(1),
                "linear_v_max": UnboundedContinuousTensorSpec(1),
                "angular_v_max": UnboundedContinuousTensorSpec(1),
                "linear_a_max": UnboundedContinuousTensorSpec(1),
                "angular_a_max": UnboundedContinuousTensorSpec(1),
                "linear_v_mean": UnboundedContinuousTensorSpec(1),
                "angular_v_mean": UnboundedContinuousTensorSpec(1),
                "linear_a_mean": UnboundedContinuousTensorSpec(1),
                "angular_a_mean": UnboundedContinuousTensorSpec(1),
            }).expand(self.num_envs).to(self.device)
        else:
            stats_spec = CompositeSpec({
                "return": UnboundedContinuousTensorSpec(1),
                "episode_len": UnboundedContinuousTensorSpec(1),
                "pos_error": UnboundedContinuousTensorSpec(1),
                "uprightness": UnboundedContinuousTensorSpec(1),
                "action_smoothness": UnboundedContinuousTensorSpec(1),
            }).expand(self.num_envs).to(self.device)

        if self.use_ctbr:
            info_spec = CompositeSpec({
                "drone_state": UnboundedContinuousTensorSpec((self.drone.n, 13), device=self.device),
                "prev_action": torch.stack([self.drone.action_spec] * self.drone.n, 0).to(self.device),
            }).expand(self.num_envs).to(self.device)
        else:
            info_spec = CompositeSpec({
                "drone_state": UnboundedContinuousTensorSpec((self.drone.n, 13), device=self.device),
            }).expand(self.num_envs).to(self.device)

        self.observation_spec["stats"] = stats_spec
        
        self.observation_spec["info"] = info_spec
        
        self.stats = stats_spec.zero()
        self.info = info_spec.zero()


    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids, self.training)
        
        pos = self.init_pos_dist.sample((*env_ids.shape, 1))
        # pos = torch.tensor([[0.81522, -0.60049, 0.96088]], device=self.device).expand(env_ids.shape[0], 1, 3)
        rpy = self.init_rpy_dist.sample((*env_ids.shape, 1))
        # rpy = torch.tensor([[-0.000, -0.012, 0.005]], device=self.device).expand(env_ids.shape[0], 1, 3)
        rot = euler_to_quaternion(rpy)
        self.drone.set_world_poses(
            pos + self.envs_positions[env_ids].unsqueeze(1), rot, env_ids
        )
        self.drone.set_velocities(self.init_vels[env_ids], env_ids)

        if self.has_payload:
            # TODO@btx0424: workout a better way 
            payload_z = self.payload_z_dist.sample(env_ids.shape)
            joint_indices = torch.tensor([self.drone._view._dof_indices["PrismaticJoint"]], device=self.device)
            self.drone._view.set_joint_positions(
                payload_z, env_indices=env_ids, joint_indices=joint_indices)
            self.drone._view.set_joint_position_targets(
                payload_z, env_indices=env_ids, joint_indices=joint_indices)
            self.drone._view.set_joint_velocities(
                torch.zeros(len(env_ids), 1, device=self.device), 
                env_indices=env_ids, joint_indices=joint_indices)
            
            payload_mass = self.payload_mass_dist.sample(env_ids.shape+(1,)) * self.drone.masses[env_ids]
            self.payload.set_masses(payload_mass, env_indices=env_ids)

        target_rpy = self.target_rpy_dist.sample((*env_ids.shape, 1))
        target_rot = euler_to_quaternion(target_rpy)
        self.target_vis.set_world_poses(orientations=target_rot, env_indices=env_ids)

        self.stats[env_ids] = 0.

        if self.use_ctbr:
            self.last_linear_v[env_ids] = torch.norm(self.init_vels[env_ids][..., :3], dim=-1)
            self.last_angular_v[env_ids] = torch.norm(self.init_vels[env_ids][..., 3:], dim=-1)
            self.last_linear_a[env_ids] = torch.zeros_like(self.last_linear_v[env_ids])
            self.last_angular_a[env_ids] = torch.zeros_like(self.last_angular_v[env_ids])
            # init prev_actions: hover
            cmd_init = 2.0 * (self.drone.throttle[env_ids]) ** 2 - 1.0
            self.info['prev_action'][env_ids, :, 3] = cmd_init.mean(dim=-1)
            self.prev_actions[env_ids] = self.info['prev_action'][env_ids].clone()



    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("agents", "action")] # For rotor command actions, not the actions output by the policy.
        self.effort = self.drone.apply_action(actions)

        if self.use_ctbr:
            self.info["prev_action"] = tensordict[("info", "prev_action")]
            self.prev_actions = self.info["prev_action"].clone()
            
            self.action_error_order1 = tensordict[("stats", "action_error_order1")].clone()
            self.stats["action_error_order1_mean"].add_(self.action_error_order1.mean(dim=-1).unsqueeze(-1))
            self.stats["action_error_order1_max"].set_(torch.max(self.stats["action_error_order1_max"], self.action_error_order1.mean(dim=-1).unsqueeze(-1)))

    def _compute_state_and_obs(self):
        self.root_state = self.drone.get_state()
        self.info["drone_state"][:] = self.root_state[..., :13]


        # relative position
        self.rpos = self.target_pos - self.root_state[..., :3]
        obs = [
            self.rpos.flatten(1).unsqueeze(1),
            # self.root_state[..., 3:7], # quat
            self.root_state[..., 7:10], # linear v
            self.root_state[..., 19:28], # rotation
        ]

        if self.time_encoding:
            t = (self.progress_buf / self.max_episode_length).unsqueeze(-1)
            obs.append(t.expand(-1, self.time_encoding_dim).unsqueeze(1))
        obs = torch.cat(obs, dim=-1)
        # add time encoding
        t = (self.progress_buf / self.max_episode_length).unsqueeze(-1)
        state = torch.concat([obs, t.expand(-1, self.time_encoding_dim).unsqueeze(1)], dim=-1).squeeze(1)

        if self.use_ctbr:
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
            
            # set last
            self.last_linear_v = self.linear_v.clone()
            self.last_angular_v = self.angular_v.clone()
            self.last_linear_a = self.linear_a.clone()
            self.last_angular_a = self.angular_a.clone()

        return TensorDict({
            "agents": {
                "observation": obs,
                "state": state,
            },
            "stats": self.stats,
            "info": self.info
        }, self.batch_size)

    def _compute_reward_and_done(self):
        # pose reward
        distance = torch.norm(self.rpos, dim=-1)

        
        # reward_pose = 1.0 / (1.0 + torch.square(self.reward_distance_scale * distance))
        # yuchao 20250221 change this reward pos error
        reward_pos = torch.exp(- distance) * self.reward_distance_scale
        reward_pos_bonus = ((distance <= 0.02) * 10).float()



        # uprightness
        tiltage = torch.abs(1 - self.drone.up[..., 2])
        reward_up = self.reward_up_weight * 0.5 / (1.0 + torch.square(tiltage))

        # spin reward, fixed z
        spin = torch.square(self.drone.vel_b[..., -1])
        reward_spin = self.reward_spin_weight * 0.5 / (1.0 + torch.square(spin))

        # effort
        # reward_effort = self.reward_effort_weight * torch.exp(-self.effort)
        not_begin_flag = (self.progress_buf > 1).unsqueeze(1)
        reward_action_smoothness = self.reward_action_smoothness_weight * torch.exp(-self.action_error_order1) * not_begin_flag.float()
        # reward_action_smoothness = self.reward_action_smoothness_weight * torch.exp(-self.drone.throttle_difference)

        reward = (
            reward_pos
            + reward_pos_bonus
            + reward_pos*(reward_up + reward_spin)
            + reward_action_smoothness
        )
        
        done = (
            (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
            | (self.drone.pos[..., 2] < 0.1)
        )

        self.stats["hover_error"].add_(-distance)
        self.stats["hover_error_ema"].lerp_(distance, (1-self.alpha))

        # bonus
        self.stats['reward_pos'].add_(reward_pos)
        self.stats['reward_spin'].add_(reward_pos * reward_spin)
        self.stats['reward_up'].add_(reward_pos * reward_up)
        self.stats["pos_bonus"].add_(reward_pos_bonus)

        ep_len = self.progress_buf.unsqueeze(-1)
        self.stats["hover_error"].div_(
            torch.where(done, ep_len, torch.ones_like(ep_len))
        )
        self.stats["pos_bonus"].div_(
            torch.where(done, ep_len, torch.ones_like(ep_len))
        )
        self.stats['action_error_order1_mean'].div_(
            torch.where(done, ep_len, torch.ones_like(ep_len))
        )
        self.stats['smoothness_mean'].div_(
            torch.where(done, ep_len, torch.ones_like(ep_len))
        )
        self.stats['reward_pos'].div_(
            torch.where(done, ep_len, torch.ones_like(ep_len))
        )
        self.stats['reward_spin'].div_(
            torch.where(done, ep_len, torch.ones_like(ep_len))
        )
        self.stats['reward_up'].div_(
            torch.where(done, ep_len, torch.ones_like(ep_len))
        )
        self.stats['reward_action_smoothness'].div_(
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
        self.stats["return"] += reward
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)

        return TensorDict(
            {
                "agents": {
                    "reward": reward.unsqueeze(-1)
                },
                "done": done,
            },
            self.batch_size,
        )
