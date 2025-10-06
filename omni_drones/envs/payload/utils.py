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


import omni.isaac.core.utils.prims as prim_utils
import omni.physx.scripts.utils as script_utils
from pxr import UsdPhysics
import omni.isaac.core.objects as objects

import omni_drones.utils.kit as kit_utils
from ..utils import create_bar

def attach_payload(
    drone_prim_path: str,
    bar_length: str,
    payload_radius: float=0.04,
    payload_mass: float=0.3
):
    bar = prim_utils.create_prim(
        prim_path=drone_prim_path + "/bar",
        prim_type="Capsule",
        translation=(0., 0., -bar_length / 2.),
        attributes={"radius": 0.01, "height": bar_length}
    )
    bar.GetAttribute('primvars:displayColor').Set([(0.8, 0.1, 0.1)])
    UsdPhysics.RigidBodyAPI.Apply(bar)
    UsdPhysics.CollisionAPI.Apply(bar)
    massAPI = UsdPhysics.MassAPI.Apply(bar)
    massAPI.CreateMassAttr().Set(0.001)

    base_link = prim_utils.get_prim_at_path(drone_prim_path + "/base_link")
    stage = prim_utils.get_current_stage()
    joint = script_utils.createJoint(stage, "D6", bar, base_link)
    joint.GetAttribute("limit:rotX:physics:low").Set(-120)
    joint.GetAttribute("limit:rotX:physics:high").Set(120)
    joint.GetAttribute("limit:rotY:physics:low").Set(-120)
    joint.GetAttribute("limit:rotY:physics:high").Set(120)
    UsdPhysics.DriveAPI.Apply(joint, "rotX")
    UsdPhysics.DriveAPI.Apply(joint, "rotY")
    joint.GetAttribute("drive:rotX:physics:damping").Set(2e-6)
    joint.GetAttribute("drive:rotY:physics:damping").Set(2e-6)
    # joint.GetAttribute('physics:excludeFromArticulation').Set(True)

    payload = objects.DynamicSphere(
        prim_path=drone_prim_path + "/payload",
        translation=(0., 0., -bar_length),
        radius=payload_radius,
        mass=payload_mass
    )
    joint = script_utils.createJoint(stage, "Fixed", bar, payload.prim)
    kit_utils.set_collision_properties(
        drone_prim_path + "/bar", contact_offset=0.02, rest_offset=0
    )
    kit_utils.set_collision_properties(
        drone_prim_path + "/payload", contact_offset=0.02, rest_offset=0
    )


# def attach_payload(
#     drone_prim_path: str,
#     bar_length: float,
#     payload_radius: float=0.04,
#     payload_mass: float=0.3
# ):

#     payload = objects.DynamicSphere(
#         prim_path=drone_prim_path + "/payload",
#         translation=(0., 0., -bar_length),
#         radius=payload_radius,
#         mass=payload_mass
#     )
#     create_bar(
#         drone_prim_path + "/bar",
#         length=bar_length,
#         from_prim=drone_prim_path + "/base_link",
#         to_prim=drone_prim_path + "/payload",
#         joint_to_attributes={}
#     )
#     kit_utils.set_collision_properties(
#         drone_prim_path + "/payload", contact_offset=0.02, rest_offset=0
#     )

