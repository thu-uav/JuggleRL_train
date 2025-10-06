import os

from typing import Dict, Optional
import torch
from torch import vmap

import hydra
from omegaconf import OmegaConf
from omni_drones import CONFIG_PATH, init_simulation_app
from tensordict import TensorDict


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg):
    OmegaConf.resolve(cfg)
    simulation_app = init_simulation_app(cfg)
    print(OmegaConf.to_yaml(cfg))

    import omni.isaac.core.utils.prims as prim_utils
    import omni_drones.utils.kit as kit_utils
    import omni_drones.utils.scene as scene_utils
    from omni.isaac.core.simulation_context import SimulationContext
    from omni_drones.controllers import LeePositionController
    from omni_drones.robots import RobotCfg
    from omni_drones.robots.drone import Crazyflie, Firefly, Hummingbird, MultirotorBase
    from omni_drones.sensors.camera import Camera, PinholeCameraCfg

    sim = SimulationContext(
        stage_units_in_meters=1.0,
        physics_dt=0.01,
        rendering_dt=0.01,
        sim_params=cfg.sim,
        backend="torch",
        device=cfg.sim.device,
    )
    n = 4

    drone_cfg = Firefly.cfg_cls()
    drone = Firefly(cfg=drone_cfg)

    translations = torch.tensor([
        [0, -1, 1.5],
        [0, 0., 1.5],
        [0, 1., 1.5],
        [0., 2., 2.5]
    ])
    drone.spawn(translations=translations)
    scene_utils.design_scene()

    camera_cfg = PinholeCameraCfg(
        sensor_tick=0,
        resolution=(960, 720),
        data_types=["rgb"],
        usd_params=PinholeCameraCfg.UsdCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
    )
    camera = Camera(camera_cfg)
    camera.spawn(
        ["/World/Camera"],
        translations=[(8, 2., 2.)],
        targets=[(0., 0., 1.75)]
    )

    sim.reset()
    drone.initialize()
    camera.initialize("/World/Camera")

    controller = drone.DEFAULT_CONTROLLER(
        g=9.81, uav_params=drone.params
    ).to(sim.device)

    control_target = torch.zeros(n, 7, device=sim.device)
    control_target[:, 0] = 0
    control_target[:, 1] = translations[:, 1]
    control_target[:, 2] = translations[:, 2]
    action = drone.action_spec.zero((n,))
    
    
    frames = []
    from tqdm import tqdm
    t = tqdm(range(2000))
    for i in t:
        if sim.is_stopped():
            break
        if not sim.is_playing():
            continue
        root_state = drone.get_state()[..., :13].squeeze(0)
        distance = torch.norm(root_state[-1, :2] - control_target[-1, :2])
        if distance < 0.05:
            control_target[-1, 1] = -control_target[-1, 1]
        action = vmap(controller)(root_state, control_target)
        drone.apply_action(action)
        sim.step(i % 2 == 0)

        if i % 2 == 0 and len(frames) < 1000:
            frame = camera.get_images()
            frames.append(frame.cpu())

    from torchvision.io import write_video

    for k, v in torch.stack(frames).cpu().items():
        for i, vv in enumerate(v.unbind(1)):
            if vv.shape[1] == 4: # rgba
                write_video(f"{k}_{i}.mp4", vv[:, :3].permute(0, 2, 3, 1), fps=50)

    simulation_app.close()


if __name__ == "__main__":
    main()
