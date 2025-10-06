from numbers import Number
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from torch import vmap
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from tensordict.utils import expand_right
from tensordict.nn import make_functional, TensorDictModule, TensorDictParams, TensorDictModuleBase
from torch.optim import lr_scheduler

from torchrl.data import (
    BoundedTensorSpec,
    CompositeSpec,
    MultiDiscreteTensorSpec,
    DiscreteTensorSpec,
    TensorSpec,
    UnboundedContinuousTensorSpec as UnboundedTensorSpec,
)

from omni_drones.utils.torchrl.env import AgentSpec
from omni_drones.utils.tensordict import print_td_shape

from ..utils import valuenorm
from ..utils.gae import compute_gae

LR_SCHEDULER = lr_scheduler._LRScheduler
from torchrl.modules import TanhNormal, IndependentNormal

from omni_drones.learning import MAPPOPolicy
from omni_drones.learning import MAPPOPolicy_mask

class MAPPOPolicy_Block_multi(object):
    """
    More specifically, PPO Policy for PSRO
    only tested in a two-agent task without RNN and actor sharing
    """

    def __init__(self, cfg, agent_spec_dict: Dict[str, AgentSpec], device="cuda") -> None:
        super().__init__()

        SecPass_agent_spec = agent_spec_dict["SecPass"]
        Att_hover_agent_spec = agent_spec_dict["Att_hover"]
        Att_agent_spec = agent_spec_dict["Att"]
        Block_hover_agent_spec = agent_spec_dict["Block_hover"]
        Block_agent_spec = agent_spec_dict["Block"]

        cfg.agent_name = "SecPass"
        self.policy_SecPass = MAPPOPolicy(cfg=cfg, agent_spec=SecPass_agent_spec, device=device)
        cfg.agent_name = "Att_hover"
        self.policy_Att_hover = MAPPOPolicy(cfg=cfg, agent_spec=Att_hover_agent_spec, device=device)
        cfg.agent_name = "Att"
        self.policy_Att = MAPPOPolicy(cfg=cfg, agent_spec=Att_agent_spec, device=device)
        cfg.agent_name = "Block_hover"
        self.policy_Block_hover = MAPPOPolicy(cfg=cfg, agent_spec=Block_hover_agent_spec, device=device)
        cfg.agent_name = "Block"
        self.policy_Block = MAPPOPolicy_mask(cfg=cfg, agent_spec=Block_agent_spec, device=device, mask_name=('stats', 'SecPass_hit'))

    def load_state_dict(self, state_dict, player: str):
        if player == "SecPass":
            self.policy_SecPass.load_state_dict(state_dict)
        elif player == "Att_hover":
            self.policy_Att_hover.load_state_dict(state_dict)
        elif player == "Att":
            self.policy_Att.load_state_dict(state_dict)
        elif player == "Block_hover":
            self.policy_Block_hover.load_state_dict(state_dict)
        elif player == "Block":
            self.policy_Block.load_state_dict(state_dict)
        
    def state_dict(self, player: str):
        if player == "SecPass":
            return self.policy_SecPass.state_dict()
        elif player == "Att_hover":
            return self.policy_Att_hover.state_dict()
        elif player == "Att":
            return self.policy_Att.state_dict()
        elif player == "Block_hover":
            return self.policy_Block_hover.state_dict()
        elif player == "Block":
            return self.policy_Block.state_dict()

    def train_op(self, tensordict: TensorDict):
        return self.policy_Block.train_op(tensordict)

    def __call__(self, tensordict: TensorDict, deterministic: bool = False):
        self.policy_SecPass(tensordict, deterministic)
        self.policy_Att_hover(tensordict, deterministic)
        self.policy_Att(tensordict, deterministic)
        self.policy_Block_hover(tensordict, deterministic)
        self.policy_Block(tensordict, deterministic)
        return tensordict