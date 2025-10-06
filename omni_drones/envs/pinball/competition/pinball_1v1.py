from omegaconf import DictConfig

import omni_drones.utils.kit as kit_utils
from omni_drones.utils.torch import euler_to_quaternion
import omni.isaac.core.objects as objects
import omni.isaac.core.materials as materials
import torch
import torch.distributions as D

from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.robots.drone import MultirotorBase
from omni_drones.views import RigidPrimView

from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import (
    UnboundedContinuousTensorSpec,
    CompositeSpec,
    DiscreteTensorSpec,
)
from omni.isaac.debug_draw import _debug_draw

from pxr import UsdShade, PhysxSchema


from typing import Dict, List

from .draw import draw_court
from .rules import determine_game_result, game_result_to_matrix, misbehave_as_lose

from carb import Float3

from omni.isaac.orbit.sensors import ContactSensorCfg, ContactSensor

from ..common import rectangular_cuboid_edges,_carb_float3_add

import logging

from omni_drones.utils.torch import (
    normalize, off_diag, quat_rotate, quat_rotate_inverse, quat_axis, symlog
)

def _carb_float3_add(a: Float3, b: Float3) -> Float3:
    return Float3(a.x + b.x, a.y + b.y, a.z + b.z)


def _vec3_format(vec3: List[float]):
    return f"({vec3[0]:.2e},{vec3[1]:.2e},{vec3[2]:.2e})"


def get_extended_rpy_dist(dist_cfg: Dict, drone_0_near_side: bool, device):
    near_low = torch.tensor(dist_cfg["low"], device=device)
    near_high = torch.tensor(dist_cfg["high"], device=device)
    far_low = near_low.clone()
    far_low[-1] = 3.1415926
    far_high = near_high.clone()
    far_high[-1] = 3.1415926

    if drone_0_near_side:
        return D.Uniform(
            low=torch.stack((near_low,far_low), dim=0),
            high=torch.stack((near_high,far_high), dim=0),
        )
    else:
        return D.Uniform(
            low=torch.stack((far_low,near_low), dim=0),
            high=torch.stack((far_high,near_high), dim=0),
        )


def get_extended_pos_dist(
    x_low: float,
    y_low: float,
    z_low: float,
    x_high: float,
    y_high: float,
    z_high: float,
    drone_0_near_side: bool,
    device,
):
    """_summary_

    Args:
        initial position distribution of drone_near_side
        device (_type_): _description_

    Returns:
        _type_: _description_
    """

    if drone_0_near_side: # drone 0 is at the near side of the court
        return D.Uniform(
            low=torch.tensor(
                [
                    [x_low, y_low, z_low],
                    [-x_high, y_low, z_low],
                ],
                device=device,
            ),
            high=torch.tensor(
                [
                    [x_high, y_high, z_high],
                    [-x_low, y_high, z_high],
                ],
                device=device,
            ),
        )
    else: # drone 0 is at the far side of the court
        return D.Uniform(
            low=torch.tensor(
                [
                    [-x_high, y_low, z_low],
                    [x_low, y_low, z_low],
                ],
                device=device,
            ),
            high=torch.tensor(
                [
                    [-x_low, y_high, z_high],
                    [x_high, y_high, z_high],
                ],
                device=device,
            ),
        )


def turn_to_obs(t: torch.Tensor, symmetric_obs: bool = False) -> torch.Tensor:
    """convert representation of drone turn to one-hot vector

    Args:
        t (torch.Tensor): (n_env,)

    Returns:
        torch.Tensor: (n_env, 2, 2)
    """
    if symmetric_obs:
        table = torch.tensor(
            [
                [
                    [1.0, 0.0],
                    [0.0, 1.0]
                ],
                [
                    [0.0, 1.0],
                    [1.0, 0.0]
                ]
            ],
            device=t.device,
        )
    else:
        table = torch.tensor(
            [
                [
                    [1.0, 0.0],
                    [1.0, 0.0]
                ],
                [
                    [0.0, 1.0],
                    [0.0, 1.0]
                ]
            ],
            device=t.device,
        )

    return table[t]


def turn_to_mask(turn: torch.Tensor) -> torch.Tensor:
    """_summary_

    Args:
        turn (torch.Tensor): (*,)

    Returns:
        torch.Tensor: (*,2)
    """
    table = torch.tensor([[True, False], [False, True]], device=turn.device)
    return table[turn]


def not_in_bounds(pos: torch.Tensor, L: float, W: float) -> torch.Tensor:
    """_summary_

    Args:
        pos (torch.Tensor): (*,3)
        L (float): _description_
        W (float): _description_

    Returns:
        torch.Tensor: (*,1)
    """
    boundary_xy = (
        torch.tensor([L / 2, W / 2],
                     device=pos.device).unsqueeze(0).unsqueeze(0)
    )  # (1,1,2)
    pos_xy = pos[..., :2].abs()  # (*,2)

    tmp = (pos_xy > boundary_xy).any(-1)  # (*,1)
    return tmp


def calculate_penalty_drone_too_near_boundary(
    drone_pos: torch.Tensor, L: float, W: float
) -> torch.Tensor:
    dist_x = L / 2 / 5
    penalty_x = ((L / 2 - drone_pos[:, :, 0].abs()) < dist_x) * 0.5  # (E,4)

    dist_y = W / 2 / 5
    penalty_y = (drone_pos[:, :, 1].abs() < dist_y).float() * 0.5 + (
        (W / 2 - drone_pos[:, :, 1].abs()) < dist_y
    ) * 0.5  # (E,4)

    return penalty_x + penalty_y  # (E,2)


def calculate_penalty_drone_pos(
    drone_pos: torch.Tensor, L: float, W: float
) -> torch.Tensor:
    """_summary_

    Args:
        drone_pos (torch.Tensor): (E,4,3)
        L (float): _description_
        W (float): _description_

    Returns:
        torch.Tensor: _description_
    """

    # 太靠近网或者后面的边界
    dist_x = L / 2 / 5
    penalty_x = (drone_pos[:, :, 0].abs() < dist_x).float() * 0.5 + (
        (L / 2 - drone_pos[:, :, 0].abs()) < dist_x).float() * 0.5  # (E,2)

    # 太靠近两条边线
    dist_y = W / 2 / 5
    penalty_y = (drone_pos[:, :, 1].abs() < dist_y).float() * 0.5 + (
        (W / 2 - drone_pos[:, :, 1].abs()) < dist_y).float() * 0.5  # (E,4)

    # 太靠近地面或者太高
    penalty_z = (1.3 - drone_pos[:, :, 2]).clamp(min=0.0) + (drone_pos[:, :, 2] - 2.0).clamp(min=0.0)

    reward = penalty_x + penalty_y + penalty_z

    return reward


def calculate_ball_hit_net(
    ball_pos: torch.Tensor, r: float, W: float, H_NET: float
) -> torch.Tensor:
    """The function is kinematically incorrect and only applicable to non-boundary cases.
    But it's efficient and very easy to implement

    Args:
        ball_pos (torch.Tensor): (E,1,3)
        r (float): radius of the ball
        W (float): width of the imaginary net
        H_NET (float): height of the imaginary net

    Returns:
        torch.Tensor: (E,1)
    """
    tmp = (
        (ball_pos[:, :, 0].abs() < 3 * r) # * 3 is to avoid the case where the ball hits the net without being reported due to simulation steps
        & (ball_pos[:, :, 1].abs() < W / 2)
        & (ball_pos[:, :, 2] < H_NET)
    )  # (E,1)
    return tmp


class PinBall1v1(IsaacEnv):
    """Two drones hit the ball adversarially.

    The net is positioned parallel to the y-axis.

    """

    def __init__(self, cfg: DictConfig, headless: bool):

        self.L: float = cfg.task.court.L
        self.W: float = cfg.task.court.W
        self.H_NET: float = cfg.task.court.H_NET # height of the net
        self.W_NET: float = cfg.task.court.W_NET # not width of the net, but the width of the net's frame

        self.ball_mass: float = cfg.task.get("ball_mass", 0.05)
        self.ball_radius: float = cfg.task.get("ball_radius", 0.03)

        self.symmetric_obs: bool = cfg.task.get("symmetric_obs", False)

        super().__init__(cfg, headless)

        self.central_env_pos = Float3(
            *self.envs_positions[self.central_env_idx].tolist()
        )
        print(f"Central env position:{self.central_env_pos}")

        self.drone.initialize()

        self.ball = RigidPrimView(
            "/World/envs/env_*/ball",
            reset_xform_properties=False,
            track_contact_forces=False,
            shape=(-1, 1),
        )
        self.ball.initialize()

        contact_sensor_cfg = ContactSensorCfg(
            prim_path="/World/envs/env_.*/ball",
        )
        self.contact_sensor: ContactSensor = contact_sensor_cfg.class_type(contact_sensor_cfg)
        self.contact_sensor._initialize_impl()

        self.init_ball_offset = torch.tensor(
            cfg.task.initial.ball_offset, device=self.device
        )

        self.drone_0_near_side = cfg.task.initial.drone_0_near_side

        self.init_on_the_spot = cfg.task.initial.get("init_on_the_spot", False)

        self.drone_xyz_dist_near_mean = (torch.tensor(cfg.task.initial.drone_xyz_dist_near.low, device=self.device) + torch.tensor(cfg.task.initial.drone_xyz_dist_near.high, device=self.device))/2

        self.init_drone_pos_dist = get_extended_pos_dist(
            *cfg.task.initial.drone_xyz_dist_near.low,
            *cfg.task.initial.drone_xyz_dist_near.high,
            drone_0_near_side=self.drone_0_near_side,
            device=self.device,
        )
        self.init_drone_rpy_dist = get_extended_rpy_dist(
            cfg.task.initial.drone_rpy_dist_near, drone_0_near_side=self.drone_0_near_side, device=self.device
        )  # unit: \pi

        if cfg.task.get(
            "control_both_courts", False
        ):
            # self.control_court_mask = torch.randn(self.num_envs, device=self.device) >0  # (E,)
            self.control_court_mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
            self.control_court_mask[ : self.num_envs // 2] = True
        else:
            self.control_court_mask = None

        # (n_envs,) 0/1
        self.turn = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)

        self.last_hit_t = torch.zeros(
            self.num_envs, 2, device=self.device, dtype=torch.int64
        )

        self._reward_drone_0 = torch.zeros(self.num_envs, device=self.device)
        self._reward_drone_1 = torch.zeros(self.num_envs, device=self.device)

        self._num_hits_drone_0 = torch.zeros(self.num_envs, device=self.device)
        self._num_hits_drone_1 = torch.zeros(self.num_envs, device=self.device)

        self.draw = _debug_draw.acquire_debug_draw_interface()

        self.ball_initial_z_vel = cfg.task.initial.get("ball_initial_z_vel", 0.0)
        # print("Ball initial z velocity:", self.ball_initial_z_vel)

        # one-hot id [E,2,2]
        self.id = torch.zeros((cfg.task.env.num_envs, 2, 2), device=self.device)
        self.id[:, 0, 0] = 1
        self.id[:, 1, 1] = 1

        if self.drone_0_near_side:
            self.anchor = torch.tensor(
                cfg.task.anchor, device=self.device)  # (2,3) original positions of two drones without any offset
        else: 
            self.anchor = torch.tensor(
                cfg.task.anchor, device=self.device)
            self.anchor[0, 0] = - self.anchor[0, 0]
            self.anchor[1, 0] = - self.anchor[1, 0]

        self.not_reset_keys_in_stats = ["actor_0_wins", "actor_1_wins", "terminated", "truncated", "done"]
        # torchrl在遇到done的时：tensordict分为root和next，会把next拿出来作为新的tensordict_2的root并且reset，但是由于共享地址，会导致之前那个tensordict也被reset
        # 见_reset_idx，所以希望在stats能保留下来，因为这几个量只有最后end时候才会生成，且判定不是按照stats里面的done，而是根据root的done判定，所以不会有问题

        self.random_turn = cfg.task.get("random_turn", False)

    def _design_scene(self):
        drone_model = MultirotorBase.REGISTRY[self.cfg.task.drone_model]
        cfg = drone_model.cfg_cls(force_sensor=self.cfg.task.force_sensor)
        self.drone: MultirotorBase = drone_model(cfg=cfg)

        material = materials.PhysicsMaterial(
            prim_path="/World/Physics_Materials/physics_material_0",
            restitution=0.8,
        )
        
        ball = objects.DynamicSphere(
            prim_path="/World/envs/env_0/ball",
            radius=self.ball_radius,
            mass=self.ball_mass,
            color=torch.tensor([1.0, 0.2, 0.2]),
            physics_material=material,
        )
        cr_api = PhysxSchema.PhysxContactReportAPI.Apply(ball.prim)
        cr_api.CreateThresholdAttr().Set(0.)

        kit_utils.create_ground_plane(
            "/World/defaultGroundPlane",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
            # color=(0.0, 100 / 256, 0.0),
        )

        # placeholders
        drone_prims = self.drone.spawn(
            translations=[
                (1.0, -1.0, 1.0),
                (1.0, 1.0, 2.0),
            ]
        )

        material = UsdShade.Material(material.prim)
        for drone_prim in drone_prims:
            collision_prim = drone_prim.GetPrimAtPath("base_link/collisions")
            binding_api = UsdShade.MaterialBindingAPI(collision_prim)
            binding_api.Bind(
                material, UsdShade.Tokens.weakerThanDescendants, "physics")

        return ["/World/defaultGroundPlane"]

    def _set_specs(self):
        drone_state_dim = self.drone.state_spec.shape[-1]  # 23

        if self.symmetric_obs:
            observation_dim = drone_state_dim + 3 + 3 + 3 + 3 + 2
        else:
            observation_dim = drone_state_dim + 3 + 3 + 3 + 3 + 2 + 2

        self.observation_spec = (
            CompositeSpec(
                {
                    "agents": {
                        "observation": UnboundedContinuousTensorSpec(
                            (2, observation_dim)
                        ),
                        # "state": UnboundedContinuousTensorSpec(state_dim),
                    }
                }
            )
            .expand(self.num_envs)
            .to(self.device)
        )
        self.action_spec = (
            CompositeSpec(
                {
                    "agents": {
                        "action": torch.stack(
                            [self.drone.action_spec] * self.drone.n, dim=0
                        ),
                    }
                }
            )
            .expand(self.num_envs)
            .to(self.device)
        )
        self.reward_spec = (
            CompositeSpec(
                {"agents": {"reward": UnboundedContinuousTensorSpec((2, 1))}})
            .expand(self.num_envs)
            .to(self.device)
        )
        self.done_spec = (
            CompositeSpec(
                {
                    "done": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
                    "terminated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
                    "truncated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
                }
            )
            .expand(self.num_envs)
            .to(self.device)
        )

        self.agent_spec["drone"] = AgentSpec(
            "drone",
            2,
            observation_key=("agents", "observation"),
            action_key=("agents", "action"),
            reward_key=("agents", "reward"),
            state_key=("agents", "state"),
        )

        _stats_spec = CompositeSpec(
            {
                "episode_len": UnboundedContinuousTensorSpec(1),
                "num_hits": UnboundedContinuousTensorSpec(1),
                "num_hits_drone_0": UnboundedContinuousTensorSpec(1),
                "num_hits_drone_1": UnboundedContinuousTensorSpec(1),
                "win_reward": UnboundedContinuousTensorSpec(1),
                "lose_reward": UnboundedContinuousTensorSpec(1),
                "drone_0_reward": UnboundedContinuousTensorSpec(1),
                "drone_1_reward": UnboundedContinuousTensorSpec(1),
                "actor_0_wins": UnboundedContinuousTensorSpec(1),
                "actor_1_wins": UnboundedContinuousTensorSpec(1),
                "terminated": UnboundedContinuousTensorSpec(1),
                "truncated": UnboundedContinuousTensorSpec(1),
                "done": UnboundedContinuousTensorSpec(1),
            }
        )
        self.stats_cfg: DictConfig = self.cfg.task.stats
        if self.stats_cfg.get("drone_height", False):
            # 0: cummulative deviation 1: count(num_hits)
            _stats_spec.set("drone_height", UnboundedContinuousTensorSpec(1))
        if self.stats_cfg.get("win_case", False):
            _stats_spec.set("drone_0_case_1", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("drone_0_case_2", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("drone_0_case_3", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("drone_0_case_4", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("drone_0_case_5", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("drone_1_case_1", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("drone_1_case_2", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("drone_1_case_3", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("drone_1_case_4", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("drone_1_case_5", UnboundedContinuousTensorSpec(1))
        if self.stats_cfg.get("drone_0_complete_reward", False):
            _stats_spec.set("drone_0_reward_rpos", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("drone_0_reward_ball_pos", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("drone_0_reward_hit", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("drone_0_reward_win", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("drone_0_penalty_pos", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("drone_0_penalty_wrong_hit", UnboundedContinuousTensorSpec(1))
        if self.stats_cfg.get("drone_1_complete_reward", False):
            _stats_spec.set("drone_1_reward_rpos", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("drone_1_reward_ball_pos", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("drone_1_reward_hit", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("drone_1_reward_win", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("drone_1_penalty_pos", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("drone_1_penalty_wrong_hit", UnboundedContinuousTensorSpec(1))

        stats_spec = _stats_spec.expand(self.num_envs).to(self.device)

        info_spec = (
            CompositeSpec(
                {
                    "drone_state": UnboundedContinuousTensorSpec((self.drone.n, 13)),
                }
            )
            .expand(self.num_envs)
            .to(self.device)
        )
        self.observation_spec["stats"] = stats_spec
        self.observation_spec["info"] = info_spec
        self.stats = stats_spec.zero()
        self.info = info_spec.zero()

    def debug_draw_region(self):

        color = [(0., 1., 0., 1.)]
        # [topleft, topright, botleft, botright]
        
        points_start, points_end = rectangular_cuboid_edges(self.L, self.W, 0.0, 3.0)
        points_start = [_carb_float3_add(p, self.central_env_pos) for p in points_start]
        points_end = [_carb_float3_add(p, self.central_env_pos) for p in points_end]
        
        colors_line = color * len(points_start)
        sizes_line = [1.] * len(points_start)
        self.draw.draw_lines(points_start,
                                 points_end, colors_line, sizes_line)

    def debug_draw_turn(self, drone_0_near_side: bool):

        turn = self.turn[self.central_env_idx]
        ori = self.envs_positions[self.central_env_idx].detach()
        if (turn.item() == 0 and drone_0_near_side) or (turn.item() == 1 and not drone_0_near_side):
            points = torch.tensor([1., 0., 0.]).to(self.device) + ori
        else:
            points = torch.tensor([-1., 0., 0.]).to(self.device) + ori
        points = [points.tolist()]
        colors = [(1, 0, 0, 1)]
        sizes = [10.]
        self.draw.clear_points()
        self.draw.draw_points(points, colors, sizes)
        logging.info(f"Central env turn:{turn.item()}")
    
    def _reset_idx(self, env_ids: torch.Tensor):
        """_summary_

        Args:
            env_ids (torch.Tensor): (n_envs_to_reset,)
        """

        # if self.central_env_idx in env_ids.tolist():
        #     print("Central environment reset!")

        self.drone._reset_idx(env_ids, self.training)

        if self.init_on_the_spot:
            drone_near_pos = self.drone_xyz_dist_near_mean.expand(len(env_ids), 1, 3)
            # drone_near_pos = torch.tensor([[3.0, 1.0, 2.0]], device=self.device).expand(len(env_ids), 1, 3)
            
            drone_far_pos = drone_near_pos.clone()
            drone_far_pos[:, 0, 0] = - drone_far_pos[:, 0, 0]
            # drone_far_pos = torch.tensor([[-3.0, -1.0, 2.0]], device=self.device).expand(len(env_ids), 1, 3)
            
            drone_near_rpy = torch.tensor([0.0, 0.0, 0.0], device=self.device).expand(len(env_ids), 1, 3)
            drone_far_rpy = drone_near_rpy.clone()
            drone_far_rpy[:, 0, 2] = 3.1415926
            
            if self.drone_0_near_side: # drone 0 is at the near side of the court  
                drone_pos = torch.cat([drone_near_pos, drone_far_pos], dim=1)
                drone_rpy = torch.cat([drone_near_rpy, drone_far_rpy], dim=1)
            else: # drone 0 is at the far side of the court
                drone_pos = torch.cat([drone_far_pos, drone_near_pos], dim=1)
                drone_rpy = torch.cat([drone_far_rpy, drone_near_rpy], dim=1)

        else:
            drone_pos = self.init_drone_pos_dist.sample(env_ids.shape)
            drone_rpy = self.init_drone_rpy_dist.sample(env_ids.shape)

        
        drone_rot = euler_to_quaternion(drone_rpy)
        self.drone.set_world_poses(
            drone_pos +
            self.envs_positions[env_ids].unsqueeze(1), drone_rot, env_ids
        )
        self.drone.set_velocities(torch.zeros(len(env_ids), 2, 6, device=self.device), env_ids)

        # ball_pos = self.init_ball_offset_dist.sample((*env_ids.shape, 1))

        if self.random_turn:
            turn = torch.randint(0, 2, (len(env_ids),), device=self.device) # randomly choose the first player
        else:
            turn = torch.zeros(len(env_ids), dtype=torch.long, device=self.device) # always player 0 starts
            # turn = torch.ones(len(env_ids), dtype=torch.long, device=self.device) # always player 1 starts
        self.turn[env_ids] = turn

        ball_pos = drone_pos[torch.arange(len(env_ids)), turn, :] + self.init_ball_offset
        ball_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(len(env_ids), 1)
        self.ball.set_world_poses(ball_pos + self.envs_positions[env_ids], ball_rot, env_ids)
        
        ball_initial_vel = torch.zeros(len(env_ids), 6, device=self.device)
        ball_initial_vel[:, 2] = self.ball_initial_z_vel
        self.ball.set_velocities(ball_initial_vel, env_ids)

        # 试一下球打到网上会是什么效果

        # ball_pos = torch.tensor([-1.0, 0.0, 1.0], device=self.device).repeat(
        #     len(env_ids), 1
        # )
        # self.ball.set_world_poses(
        #     ball_pos + self.envs_positions[env_ids], ball_rot, env_ids
        # )
        # ball_vel = torch.tensor(
        #     [3.0, 0.0, 1.0, 0.0, 0.0, 0.0], device=self.device
        # ).repeat(len(env_ids), 1)
        # self.ball.set_velocities(ball_vel, env_ids)

        # fix the mass now
        ball_masses = torch.ones_like(env_ids) * self.ball_mass
        self.ball.set_masses(ball_masses, env_ids)

        self.last_hit_t[env_ids] = -100

        if (env_ids == self.central_env_idx).any() and self._should_render(0):
            self.draw.clear_lines()

            # point_list_1, point_list_2, colors, sizes = draw_court(
            #     self.W, self.L, self.H_NET, self.W_NET
            # )
            # point_list_1 = [
            #     _carb_float3_add(p, self.central_env_pos) for p in point_list_1
            # ]
            # point_list_2 = [
            #     _carb_float3_add(p, self.central_env_pos) for p in point_list_2
            # ]
            # self.draw.draw_lines(point_list_1, point_list_2, colors, sizes)

            logging.info("Reset central environment")
            self.debug_draw_region()
            self.debug_draw_turn(self.drone_0_near_side)

        self._reward_drone_0[env_ids] = 0.0
        self._reward_drone_1[env_ids] = 0.0
        self._num_hits_drone_0[env_ids] = 0.0
        self._num_hits_drone_1[env_ids] = 0.0
        
        # some stats keys will not reset
        stats_list = []
        for i in self.not_reset_keys_in_stats:
            stats_list.append(self.stats[i][env_ids].clone())
        self.stats[env_ids] = 0.0
        for i, key in enumerate(self.not_reset_keys_in_stats):
            self.stats[key][env_ids] = stats_list[i]

    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions: torch.Tensor = tensordict[("agents", "action")]
        if self.control_court_mask is not None:
            # TODO: I think this is wrong, need to be checked
            # actions.shape: (E, 2, action_dim)
            assert len(actions.shape) == 3
            assert actions.shape[1] == 2
            flipped_actions = actions[:, [1, 0], :]
            mask = self.control_court_mask.float().unsqueeze(-1).unsqueeze(-1)  # (E,1,1)
            actions = actions * (1.0 - mask) + flipped_actions * mask
            self.effort = self.drone.apply_action(actions)
        else:
            self.effort = self.drone.apply_action(actions)
    
    def _post_sim_step(self, tensordict: TensorDictBase):
        self.contact_sensor.update(self.dt)

    def _compute_state_and_obs(self):
        # clone here
        self.root_state = self.drone.get_state()
        # pos(3), quat(4), vel, omega
        self.info["drone_state"][:] = self.root_state[..., :13]
        self.ball_pos, _ = self.get_env_poses(self.ball.get_world_poses())
        self.ball_vel = self.ball.get_velocities()[..., :3]

        # if self.ball_vel[0, 0, 0] != 0:
        #     import pdb; pdb.set_trace()

        if self.symmetric_obs:

            root_state = self.root_state.clone()

            pos = root_state[..., :3]
            pos[:, 1, :2] = - pos[:, 1, :2]

            rot = root_state[..., 3:7]
            rot[:, 0, :] = torch.where((rot[:, 0, 0] < 0).unsqueeze(-1), -rot[:, 0, :], rot[:, 0, :]) # make sure w of (w,x,y,z) is positive

            def quaternion_multiply(q1, q2):
                assert q1.shape == q2.shape and q1.shape[-1] == 4
                w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
                w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
                w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
                x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
                y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
                z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
                return torch.stack([w, x, y, z], dim=-1)
            # q_m: 围绕z轴180度旋转的四元数
            q_m = torch.tensor([[0, 0, 0, 1]] * rot.shape[0], device=rot.device)
            rot[:, 1, :] = quaternion_multiply(q_m, rot[:, 1, :])
            rot[:, 1, :] = torch.where((rot[:, 1, 0] < 0).unsqueeze(-1), -rot[:, 1, :], rot[:, 1, :]) # make sure w of (w,x,y,z) is positive

            vel = root_state[..., 7:10]
            vel[:, 1, :2] = - vel[:, 1, :2]

            omega = root_state[..., 10:13]
            omega[:, 1, :2] = - omega[:, 1, :2]

            heading = root_state[..., 13:16]
            heading[:, 1, :] = quat_axis(rot[:, 1, :], axis=0)
        
            up = root_state[..., 16:19]
            up[:, 1, :] = quat_axis(rot[:, 1, :], axis=2)

            rotors = root_state[..., 19:23]

            # relative position and heading
            self.rpos_ball = self.drone.pos - self.ball_pos
            rpos_ball = self.rpos_ball.clone()
            rpos_ball[:, 1, :2] = - rpos_ball[:, 1, :2]

            self.rpos_drone = torch.stack(
                [
                    # [..., drone_id, [x, y, z]]
                    self.drone.pos[..., 1, :] - self.drone.pos[..., 0, :],
                    self.drone.pos[..., 0, :] - self.drone.pos[..., 1, :],
                ],
                dim=1,
            )  # (E,2,3)
            rpos_drone = self.rpos_drone.clone()
            rpos_drone[:, 1, :2] = - rpos_drone[:, 1, :2]

            rpos_anchor = self.drone.pos - self.anchor  # (E,2,3)
            rpos_anchor[:, 1, :2] = - rpos_anchor[:, 1, :2]

            # if self.ball_vel[0, 0, 0] != 0:
            #     import pdb; pdb.set_trace()
            # ball_vel = self.ball_vel.clone().expand(-1, 2, 3)
            ball_vel = self.ball_vel.clone().repeat(1, 2, 1)
            ball_vel[:, 1, :2] = - ball_vel[:, 1, :2]

            obs = [
                pos, # (E,2,3)
                rot, # (E,2,4)
                vel, # (E,2,3)
                omega, # (E,2,3)
                heading, # (E,2,3)
                up, # (E,2,3)
                rotors, # (E,2,4)
                rpos_anchor,  # [E,2,3]
                rpos_drone,  # [E,2,3]
                rpos_ball,  # [E,2,3]
                ball_vel,  # [E,2,3]
                turn_to_obs(self.turn, symmetric_obs=True), # [E,2,2]
                
                # self.id, # [E,2,2]
            ]
            obs = torch.cat(obs, dim=-1)

        else:
            # relative position and heading
            self.rpos_ball = self.drone.pos - self.ball_pos

            self.rpos_drone = torch.stack(
                [
                    # [..., drone_id, [x, y, z]]
                    self.drone.pos[..., 1, :] - self.drone.pos[..., 0, :],
                    self.drone.pos[..., 0, :] - self.drone.pos[..., 1, :],
                ],
                dim=1,
            )  # (E,2,3)

            rpos_anchor = self.drone.pos - self.anchor  # (E,2,3)

            obs = [
                self.root_state, # (E,2,23)
                rpos_anchor,  # [E,2,3]
                self.rpos_drone,  # [E,2,3]
                self.rpos_ball,  # [E,2,3]
                self.ball_vel.expand(-1, 2, 3),  # [E,2,3]
                turn_to_obs(self.turn), # [E,2,2]
                self.id, # [E,2,2]
            ]
            obs = torch.cat(obs, dim=-1)

            # TODO: I think this is wrong, need to be checked
            if self.control_court_mask is not None:
                flipped_obs = obs[:, [1, 0], :]
                mask = self.control_court_mask.float().unsqueeze(-1).unsqueeze(-1)  # (E,1,1)
                obs = obs * (1.0 - mask) + flipped_obs * mask

        return TensorDict(
            {
                "agents": {
                    "observation": obs,
                },
                "stats": self.stats, # reset_id的时候需要保留上一个step没有reset的环境的stats
                "info": self.info,
            },
            self.num_envs,
        )

    def _compute_reward_and_done(self):
        # drone hits ball
        # ball_contact_forces_0 = self.ball.get_net_contact_forces()
        ball_contact_forces = self.contact_sensor.data.net_forces_w # (E,1,3)

        which_drone: torch.Tensor = self.rpos_ball.norm(p=2, dim=-1).argmin(dim=1, keepdim=True)  # (E,1)

        is_close = (self.rpos_ball[torch.arange(self.num_envs, device=which_drone.device), which_drone.squeeze(-1)].norm(dim=-1, keepdim=True) < 0.2)
        
        # hit_0 = ball_contact_forces.any(-1)
        hit = ball_contact_forces.any(-1) # (E,1)

        # if hit_0 != hit:
        #     import pdb; pdb.set_trace()
        
        drone_last_hit_t = self.last_hit_t[
            torch.arange(self.num_envs, device=which_drone.device),
            which_drone.squeeze(-1),
        ].unsqueeze(-1)  # (E,1) 当前距离球较近的无人机上次击球的时刻
        
        # ball_hit_drone = hit & (self.progress_buf.unsqueeze(-1) - drone_last_hit_t > 3)
        ball_hit_drone = hit
        
        new_last_hit_t = torch.where(
            hit, self.progress_buf.unsqueeze(-1), drone_last_hit_t
        )  # (E,1)
        new_last_hit_t = (
            new_last_hit_t
            * (
                which_drone == torch.arange(
                    2, device=which_drone.device).unsqueeze(0)
            ).long()
        ) + self.last_hit_t * (
            which_drone != torch.arange(
                2, device=which_drone.device).unsqueeze(0)
        ).long()
        self.last_hit_t = new_last_hit_t

        # ball hits net
        ball_hit_net = calculate_ball_hit_net(self.ball_pos, self.ball_radius, self.W, self.H_NET)

        switch_turn: torch.Tensor = ball_hit_drone & (self.turn.unsqueeze(-1) == which_drone)  # (E,1)
        wrong_hit: torch.Tensor = ball_hit_drone & (self.turn.unsqueeze(-1) != which_drone)  # (E,1)

        self.game_reward_coef = 5.0
        self.hit_reward_coef = 1.0
        self.rpos_reward_coef = 0.05
        self.drone_pos_reward_coef = 0.2
        self.ball_reward_coef = 0.1

        penalty_wrong_hit = turn_to_mask(self.turn).float() * wrong_hit.float() * self.hit_reward_coef # (E,2)

        # 约束无人机的位置不要太离谱
        penalty_pos = self.drone_pos_reward_coef * calculate_penalty_drone_pos(self.drone.pos, self.L, self.W) # (E,2)

        # heuristic 让无人机倾向于击球
        reward_rpos =  self.rpos_reward_coef * (
            turn_to_mask(self.turn).float() / (1 + torch.norm(self.rpos_ball[..., :2], p=2, dim=-1))
        )  # (E,2)

        # 击球后球应该别太低，往对面半场飞
        if self.drone_0_near_side:
            _ball_towards_opponent = (self.turn == 0).unsqueeze(-1) & (self.ball_vel[:, :, 0] > 0) | (
                self.turn == 1).unsqueeze(-1) & (self.ball_vel[:, :, 0] < 0)
        else:
            _ball_towards_opponent = (self.turn == 0).unsqueeze(-1) & (self.ball_vel[:, :, 0] < 0) | (
                self.turn == 1).unsqueeze(-1) & (self.ball_vel[:, :, 0] > 0)
        reward_ball_pos = self.ball_reward_coef * (
            (~turn_to_mask(self.turn)) & (self.stats["num_hits"] != 0)).float() * (
            (self.ball_pos[:, :, 2] > 1.0).float() * 0.5 + _ball_towards_opponent.float() * 0.5
        ) 

        reward_hit = turn_to_mask(self.turn).float() * switch_turn.float() * self.hit_reward_coef # (E,2)

        self._num_hits_drone_0.add_((self.turn == 0).float() * switch_turn.squeeze(-1).float())
        self._num_hits_drone_1.add_((self.turn == 1).float() * switch_turn.squeeze(-1).float())
        self.stats["num_hits_drone_0"] = self._num_hits_drone_0.unsqueeze(-1)
        self.stats["num_hits_drone_1"] = self._num_hits_drone_1.unsqueeze(-1)

        self.turn = (self.turn + switch_turn.squeeze(-1).long()) % 2 # update new turn
        if self._should_render(0) and switch_turn[self.central_env_idx].item():
            self.debug_draw_turn(self.drone_0_near_side)

        game_result, drone_0_case_1, drone_0_case_2, drone_0_case_3, drone_0_case_4, drone_0_case_5, drone_1_case_1, drone_1_case_2, drone_1_case_3, drone_1_case_4, drone_1_case_5 = determine_game_result(
            self.ball_pos, self.drone.pos, self.turn, ball_hit_net, self.L, self.W, wrong_hit, self.stats["num_hits"]
        )  # (E,1)

        if self._should_render(0) and (
            drone_0_case_1[self.central_env_idx].item() | 
            drone_0_case_2[self.central_env_idx].item() | 
            drone_0_case_3[self.central_env_idx].item() |
            drone_0_case_4[self.central_env_idx].item() |
            drone_0_case_5[self.central_env_idx].item() |
            drone_1_case_1[self.central_env_idx].item() | 
            drone_1_case_2[self.central_env_idx].item() | 
            drone_1_case_3[self.central_env_idx].item() |
            drone_1_case_4[self.central_env_idx].item() |
            drone_1_case_5[self.central_env_idx].item()
        ):
            if game_result[self.central_env_idx].item() == 0:
                logging.info("Draw!")
            elif game_result[self.central_env_idx].item() == 1:
                logging.info("Drone 0 wins!")
            elif game_result[self.central_env_idx].item() == -1:
                logging.info("Drone 1 wins!")

            if drone_0_case_1[self.central_env_idx].item():
                logging.info("Case 1: The ball lands on the opposite side of the court.")
            if drone_0_case_2[self.central_env_idx].item():
                logging.info("Case 2: The opponent hits the ball out of bounds.")
            if drone_0_case_3[self.central_env_idx].item():
                logging.info("Case 3: The opponent hits the ball into the net.")
            if drone_0_case_4[self.central_env_idx].item():
                logging.info("Case 4: The opponent hits the ball twice consecutively.")
            if drone_0_case_5[self.central_env_idx].item():
                logging.info("Case 5: The opponent flies too low.")

            if drone_1_case_1[self.central_env_idx].item():
                logging.info("Case 1: The ball lands on the opposite side of the court.")
            if drone_1_case_2[self.central_env_idx].item():
                logging.info("Case 2: The opponent hits the ball out of bounds.")
            if drone_1_case_3[self.central_env_idx].item():
                logging.info("Case 3: The opponent hits the ball into the net.")
            if drone_1_case_4[self.central_env_idx].item():
                logging.info("Case 4: The opponent hits the ball twice consecutively.")
            if drone_1_case_5[self.central_env_idx].item():
                logging.info("Case 5: The opponent flies too low.")

        if self.stats_cfg.get("win_case", False):
            self.stats["drone_0_case_1"].add_(drone_0_case_1.float())
            self.stats["drone_0_case_2"].add_(drone_0_case_2.float())
            self.stats["drone_0_case_3"].add_(drone_0_case_3.float())
            self.stats["drone_0_case_4"].add_(drone_0_case_4.float())
            self.stats["drone_0_case_5"].add_(drone_0_case_5.float())
            self.stats["drone_1_case_1"].add_(drone_1_case_1.float())
            self.stats["drone_1_case_2"].add_(drone_1_case_2.float())
            self.stats["drone_1_case_3"].add_(drone_1_case_3.float())
            self.stats["drone_1_case_4"].add_(drone_1_case_4.float())
            self.stats["drone_1_case_5"].add_(drone_1_case_5.float())
        
        # drone_misbehave = (
        #     (self.drone.pos[..., 2] < 0.3)
        #     | (self.drone.pos[..., 2] > 2.5)
        #     | (self.drone.pos[..., 0].abs() < 0.2)
        #     | not_in_bounds(self.drone.pos, self.L, self.W)
        # )  # (E,2)
        # penalty_drone_misbehave = drone_misbehave.float() * self.drone_misbehave_reward_coef
        # game_result = torch.where(game_result == 0, misbehave_as_lose(drone_misbehave), game_result)  # (E,1)

        reward_win = game_result_to_matrix(game_result) * self.game_reward_coef  # (E,2)

        step_reward = reward_rpos + reward_ball_pos
        conditional_reward = reward_hit - penalty_pos
        end_reward = reward_win - penalty_wrong_hit
        
        reward = step_reward + conditional_reward + end_reward

        self._reward_drone_0.add_(reward[:, 0])
        self._reward_drone_1.add_(reward[:, 1])

        if self.stats_cfg.get("drone_0_complete_reward", False):
            self.stats["drone_0_reward_rpos"].add_(reward_rpos[:, 0].unsqueeze(-1))
            self.stats["drone_0_reward_ball_pos"].add_(reward_ball_pos[:, 0].unsqueeze(-1))
            self.stats["drone_0_reward_hit"].add_(reward_hit[:, 0].unsqueeze(-1))
            self.stats["drone_0_reward_win"].add_(reward_win[:, 0].unsqueeze(-1))
            self.stats["drone_0_penalty_pos"].sub_(penalty_pos[:, 0].unsqueeze(-1))
            self.stats["drone_0_penalty_wrong_hit"].sub_(penalty_wrong_hit[:, 0].unsqueeze(-1))
        if self.stats_cfg.get("drone_1_complete_reward", False):
            self.stats["drone_1_reward_rpos"].add_(reward_rpos[:, 1].unsqueeze(-1))
            self.stats["drone_1_reward_ball_pos"].add_(reward_ball_pos[:, 1].unsqueeze(-1))
            self.stats["drone_1_reward_hit"].add_(reward_hit[:, 1].unsqueeze(-1))
            self.stats["drone_1_reward_win"].add_(reward_win[:, 1].unsqueeze(-1))
            self.stats["drone_1_penalty_pos"].sub_(penalty_pos[:, 1].unsqueeze(-1))
            self.stats["drone_1_penalty_wrong_hit"].sub_(penalty_wrong_hit[:, 1].unsqueeze(-1))

        self.stats["drone_0_reward"] = self._reward_drone_0.unsqueeze(-1)
        self.stats["drone_1_reward"] = self._reward_drone_1.unsqueeze(-1)

        self.stats["win_reward"] = (game_result == -1).float() * self._reward_drone_1.unsqueeze(-1)
        self.stats["win_reward"] += (game_result == 1).float() * self._reward_drone_0.unsqueeze(-1)

        self.stats["lose_reward"] = (game_result == -1).float() * self._reward_drone_0.unsqueeze(-1)
        self.stats["lose_reward"] += (game_result == 1).float() * self._reward_drone_1.unsqueeze(-1)

        if self.control_court_mask is not None:
            # TODO: I think this is wrong, need to be checked
            mask = self.control_court_mask.float().unsqueeze(-1)  # (E,1)
            self.stats["actor_0_wins"] = (game_result == 1).float() * (1.0 - mask) + (
                game_result == -1
            ).float() * mask
            self.stats["actor_1_wins"] = (game_result == -1).float() * (1.0 - mask) + (
                game_result == 1
            ).float() * mask
        else:
            self.stats["actor_0_wins"] = (game_result == 1).float()
            self.stats["actor_1_wins"] = (game_result == -1).float()

        # ball_misbehave = (
        #     (self.ball_pos[..., 2] < 0.3)
        #     | ball_hit_net
        #     | not_in_bounds(self.ball_pos, self.L, self.W)
        # )  # (E,1)

        truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
        terminated = (game_result != 0)
        done = truncated | terminated # [E, 1]

        self.stats["truncated"][:] = truncated.float()
        self.stats["terminated"][:] = terminated.float()
        self.stats["done"][:] = done.float()
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)
        self.stats["num_hits"].add_(switch_turn.float())

        if self.stats_cfg.get("drone_height", False):
            self.stats["drone_height"] = self.drone.pos[..., 2].mean(dim=-1, keepdim=True)

        if self.control_court_mask is not None:
            # TODO: I think this is wrong, need to be checked
            flipped_reward = reward[:, [1, 0]]
            mask = self.control_court_mask.float().unsqueeze(-1)  # (E,1)
            reward = reward * (1.0 - mask) + flipped_reward * mask

        # if self._should_render(0):
        #     if self.ball_pos[self.central_env_idx, 0, 0] < 0:
        #         p = Float3(-self.L/2, 0, 0)
        #     else:
        #         p = Float3(self.L/2, 0, 0)
        #     self.draw.draw_points([_carb_float3_add(p, self.central_env_pos),], [
        #         (1, 0, 0, 1),
        #     ], [10,])
        
        return TensorDict(
            {
                "agents": {"reward": reward.unsqueeze(-1)},
                "stats": self.stats, # 否则isaac_env里的stats会滞后trainevery个steps
                "done": done,
                "terminated": terminated,
                "truncated": truncated,
            },
            self.batch_size,
        )
