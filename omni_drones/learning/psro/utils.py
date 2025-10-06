import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictParams
from typing import Union, Callable, List, Optional
from torch import vmap
import os
import numpy as np
import contextlib
import copy
from torch import nn

class uniform_policy(nn.Module):
    def __init__(self):
        super(uniform_policy, self).__init__()
    
    def forward(self, tensordict: TensorDict):
        observation: torch.Tensor = tensordict['agents', 'observation'] # [E, 1, 33]
        
        action_dim = 4
        action_shape = observation.shape[:-1] + (action_dim,)
        action = 2 * torch.rand(action_shape, device=observation.device) - 1  # [E, 1, 4]

        action_log_prob = torch.log(torch.ones(size=action_shape, device=observation.device) / 2)  # [E, 1, 4]
        action_log_prob = torch.sum(action_log_prob, dim=-1, keepdim=True)  # [E, 1, 1]

        action_entropy = torch.ones(size=action_shape, device=observation.device) * torch.log(torch.tensor(2.0, device=observation.device))  # [E, 1, 4]
        action_entropy = torch.sum(action_entropy, dim=-1, keepdim=True)  # [E, 1, 1]

        tensordict.set(("agents", "action"), action)
        tensordict.set("drone.action_logp", action_log_prob)
        tensordict.set("drone.action_entropy", action_entropy)

        return tensordict
    
# def _uniform_policy(tensordict: TensorDict):
#     """
#     Uniform policy for initialization.
#     Each action dim is i.i.d. uniformly sampled from [-1, 1].
#     """

#     observation: torch.Tensor = tensordict['agents', 'observation'] # [E, 1, 33]
    
#     action_dim = 4
#     action_shape = observation.shape[:-1] + (action_dim,)
#     action = 2 * torch.rand(action_shape, device=observation.device) - 1  # [E, 1, 4]

#     action_log_prob = torch.log(torch.ones(size=action_shape, device=observation.device) / 2)  # [E, 1, 4]
#     action_log_prob = torch.sum(action_log_prob, dim=-1, keepdim=True)  # [E, 1, 1]

#     action_entropy = torch.ones(size=action_shape, device=observation.device) * torch.log(torch.tensor(2.0, device=observation.device))  # [E, 1, 4]
#     action_entropy = torch.sum(action_entropy, dim=-1, keepdim=True)  # [E, 1, 1]

#     tensordict.set(("agents", "action"), action)
#     tensordict.set("drone.action_logp", action_log_prob)
#     tensordict.set("drone.action_entropy", action_entropy)

#     return tensordict


_policy_t = Callable[[TensorDict], TensorDict]


class Population:
    def __init__(
        self,
        dir: str,
        module: TensorDictModule, # by reference
        initial_policy: Union[uniform_policy, TensorDictParams] = uniform_policy(),
        device = "cuda",
    ):
        self.dir = dir
        os.makedirs(self.dir, exist_ok=True)
        self._module_cnt = 0 # How many policies have been saved
        self._module = module # actor: assume all modules are homogeneous
        self._idx = -1
        self.device = device

        self.policy_sets: List[Union[_policy_t, int]] = []

        # policy 0
        if not isinstance(initial_policy, (TensorDictParams, TensorDict)):
            self.policy_sets.append(initial_policy) # initial policy is a function/class
            self._module_cnt += 1
        else:
            self.add(initial_policy)

        self._func = None # current used policy
        self._params = None

        self.sample(meta_policy=np.array([1.0]))

    def __len__(self) -> int:
        return len(self.policy_sets)

    def add(self, params: TensorDictParams):
        if len(self.policy_sets) == 1 and isinstance(self.policy_sets[0], Callable): # the first policy is a random policy
            self._module_cnt = 1
            torch.save(params, os.path.join(self.dir, f"{self._module_cnt}.pt"))
            self.policy_sets = [self._module_cnt]
            self._idx = -1
        else:
            self._module_cnt += 1
            torch.save(params, os.path.join(self.dir, f"{self._module_cnt}.pt"))
            self.policy_sets.append(self._module_cnt)
        self.set_latest_policy() # set the new policy as the current policy

    def _set_policy(self, index: int, ):
        if self._idx == index:
            return
        
        if not isinstance(self.policy_sets[index], int):
            self._func = self.policy_sets[index]
        else:
            assert self._module is not None
            self._params=torch.load(os.path.join(self.dir, f"{self.policy_sets[index]}.pt"))
            # self._func = lambda tensordict:vmap(
            #     self._module, in_dims=(1, 0), out_dims=1, randomness="different"
            # )(tensordict, self._params, deterministic=False)
            self._func = lambda tensordict:vmap(
                self._module, in_dims=(1, 0), out_dims=1, randomness="error"
            )(tensordict, self._params, deterministic=True)
        
        self._idx = index
    
    def set_latest_policy(self):
        self._set_policy(self._module_cnt - 1)

    def set_second_latest_policy(self):
        self._set_policy(self._module_cnt - 2)

    def sample(self, meta_policy: np.array):
        # import pdb; pdb.set_trace()
        if len(meta_policy) == len(self.policy_sets):
            self._set_policy(np.random.choice(len(self.policy_sets), p=meta_policy))
        elif len(meta_policy) == len(self.policy_sets) - 1: # the population of one player is updated with a new policy while the population of the other players remains the same
            prob = np.append(meta_policy, 0.0)
            self._set_policy(np.random.choice(len(self.policy_sets), p=prob))
        else:
            raise ValueError("Invalid meta_policy")

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        tensordict = tensordict.to(self.device)
        return self._func(tensordict)

    def set_behavioural_strategy(self, index: int):
        self._set_policy(index)
        
    def make_behavioural_strategy(self, index: int) -> _policy_t:
        if not isinstance(self.policy_sets[index], int):
            return self.policy_sets[index]
        
        _params=torch.load(os.path.join(self.dir, f"{self.policy_sets[index]}.pt"))
    
        def _strategy(tensordict: TensorDict) -> TensorDict:
            tensordict = tensordict.to(self.device)
            return vmap(
                self._module, in_dims=(1, 0), out_dims=1, randomness="error"
            )(tensordict, _params, deterministic=True)

        return _strategy