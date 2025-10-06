from omni_drones.utils.torch import euler_to_quaternion,quat_rotate
import torch
import torch.distributions as D
from collections import defaultdict

def fun1(p:torch.Tensor,q:torch.Tensor)->torch.Tensor:
    tmp=torch.sum(p*q)
    return 2*tmp*tmp-1

def fun2(p:torch.Tensor,q:torch.Tensor)->torch.Tensor:
    v=torch.zeros(3,)
    v[2]=1.0
    v=v.unsqueeze(0)
    p=quat_rotate(p.unsqueeze(0),v)
    q=quat_rotate(q.unsqueeze(0),v)
    tmp=p*q
    return torch.sum(tmp)


init_drone_rpy_dist = D.Uniform(
            torch.tensor([-.1, -.1, 0.]) * torch.pi,
            torch.tensor([0.1, 0.1, 2]) * torch.pi
        )

a=init_drone_rpy_dist.sample()
b=init_drone_rpy_dist.sample()
print(a,b)
a=euler_to_quaternion(a)
b=euler_to_quaternion(b)
print(a,b)
print(fun1(a,b))
print(fun2(a,b))