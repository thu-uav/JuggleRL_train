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

class MAPPOPolicy_Att_multi_hover(object):

    def __init__(self, cfg, agent_spec_dict: Dict[str, AgentSpec], device="cuda") -> None:
        super().__init__()

        Opp_SecPass_agent_spec = agent_spec_dict["Opp_SecPass"]
        Opp_SecPass_hover_agent_spec = agent_spec_dict["Opp_SecPass_hover"]

        Opp_Att_goto_agent_spec = agent_spec_dict["Opp_Att_goto"]
        Opp_Att_agent_spec = agent_spec_dict["Opp_Att"]
        Opp_Att_hover_agent_spec = agent_spec_dict["Opp_Att_hover"]

        FirstPass_goto_agent_spec = agent_spec_dict["FirstPass_goto"]
        FirstPass_agent_spec = agent_spec_dict["FirstPass"]
        FirstPass_hover_agent_spec = agent_spec_dict["FirstPass_hover"]

        SecPass_goto_agent_spec = agent_spec_dict["SecPass_goto"]
        SecPass_agent_spec = agent_spec_dict["SecPass"]
        SecPass_hover_agent_spec = agent_spec_dict["SecPass_hover"]

        Att_goto_agent_spec = agent_spec_dict["Att_goto"]
        Att_agent_spec = agent_spec_dict["Att"]
        Att_hover_agent_spec = agent_spec_dict["Att_hover"]

        cfg.agent_name = "Opp_SecPass"
        self.policy_Opp_SecPass = MAPPOPolicy(cfg=cfg, agent_spec=Opp_SecPass_agent_spec, device=device)
        cfg.agent_name = "Opp_SecPass_hover"
        self.policy_Opp_SecPass_hover = MAPPOPolicy(cfg=cfg, agent_spec=Opp_SecPass_hover_agent_spec, device=device)

        cfg.agent_name = "Opp_Att_goto"
        self.policy_Opp_Att_goto = MAPPOPolicy(cfg=cfg, agent_spec=Opp_Att_goto_agent_spec, device=device)
        cfg.agent_name = "Opp_Att"
        self.policy_Opp_Att = MAPPOPolicy(cfg=cfg, agent_spec=Opp_Att_agent_spec, device=device)
        cfg.agent_name = "Opp_Att_hover"
        self.policy_Opp_Att_hover = MAPPOPolicy(cfg=cfg, agent_spec=Opp_Att_hover_agent_spec, device=device)
        
        cfg.agent_name = "FirstPass_goto"
        self.policy_FirstPass_goto = MAPPOPolicy(cfg=cfg, agent_spec=FirstPass_goto_agent_spec, device=device)
        cfg.agent_name = "FirstPass"
        self.policy_FirstPass = MAPPOPolicy(cfg=cfg, agent_spec=FirstPass_agent_spec, device=device)
        cfg.agent_name = "FirstPass_hover"
        self.policy_FirstPass_hover = MAPPOPolicy(cfg=cfg, agent_spec=FirstPass_hover_agent_spec, device=device)

        cfg.agent_name = "SecPass_goto"
        self.policy_SecPass_goto = MAPPOPolicy(cfg=cfg, agent_spec=SecPass_goto_agent_spec, device=device)
        cfg.agent_name = "SecPass"
        self.policy_SecPass = MAPPOPolicy(cfg=cfg, agent_spec=SecPass_agent_spec, device=device)
        cfg.agent_name = "SecPass_hover"
        self.policy_SecPass_hover = MAPPOPolicy(cfg=cfg, agent_spec=SecPass_hover_agent_spec, device=device)

        cfg.agent_name = "Att_goto"
        self.policy_Att_goto = MAPPOPolicy(cfg=cfg, agent_spec=Att_goto_agent_spec, device=device)
        cfg.agent_name = "Att"
        self.policy_Att = MAPPOPolicy(cfg=cfg, agent_spec=Att_agent_spec, device=device)
        cfg.agent_name = "Att_hover"
        self.policy_Att_hover = MAPPOPolicy_mask(cfg=cfg, agent_spec=Att_hover_agent_spec, device=device, mask_name=('stats', 'Att_hit'))


    def load_state_dict(self, state_dict, player: str):
        if player == "Opp_SecPass":
            self.policy_Opp_SecPass.load_state_dict(state_dict)
        elif player == "Opp_SecPass_hover":
            self.policy_Opp_SecPass_hover.load_state_dict(state_dict)

        elif player == "Opp_Att_goto":
            self.policy_Opp_Att_goto.load_state_dict(state_dict)
        elif player == "Opp_Att":
            self.policy_Opp_Att.load_state_dict(state_dict)
        elif player == "Opp_Att_hover":
            self.policy_Opp_Att_hover.load_state_dict(state_dict)
        
        elif player == "FirstPass_goto":
            self.policy_FirstPass_goto.load_state_dict(state_dict)
        elif player == "FirstPass":
            self.policy_FirstPass.load_state_dict(state_dict)
        elif player == "FirstPass_hover":
            self.policy_FirstPass_hover.load_state_dict(state_dict)

        elif player == "SecPass_goto":
            self.policy_SecPass_goto.load_state_dict(state_dict)
        elif player == "SecPass":
            self.policy_SecPass.load_state_dict(state_dict)
        elif player == "SecPass_hover":
            self.policy_SecPass_hover.load_state_dict(state_dict)

        elif player == "Att_goto":
            self.policy_Att_goto.load_state_dict(state_dict)
        elif player == "Att":
            self.policy_Att.load_state_dict(state_dict)
        elif player == "Att_hover":
            self.policy_Att_hover.load_state_dict(state_dict)
        
        else:
            raise ValueError("player not found")
        
    def state_dict(self, player: str):
        if player == "Opp_SecPass":
            return self.policy_Opp_SecPass.state_dict()
        elif player == "Opp_SecPass_hover":
            return self.policy_Opp_SecPass_hover.state_dict()

        elif player == "Opp_Att_goto":
            return self.policy_Opp_Att_goto.state_dict()
        elif player == "Opp_Att":
            return self.policy_Opp_Att.state_dict()
        elif player == "Opp_Att_hover":
            return self.policy_Opp_Att_hover.state_dict()

        elif player == "FirstPass_goto":
            return self.policy_FirstPass_goto.state_dict()
        elif player == "FirstPass":
            return self.policy_FirstPass.state_dict()
        elif player == "FirstPass_hover":
            return self.policy_FirstPass_hover.state_dict()
        
        elif player == "SecPass_goto":
            return self.policy_SecPass_goto.state_dict()
        elif player == "SecPass":
            return self.policy_SecPass.state_dict()
        elif player == "SecPass_hover":
            return self.policy_SecPass_hover.state_dict()

        elif player == "Att_goto":
            return self.policy_Att_goto.state_dict()
        elif player == "Att":
            return self.policy_Att.state_dict()
        elif player == "Att_hover":
            return self.policy_Att_hover.state_dict()
        
        else:
            raise ValueError("player not found")

    def train_op(self, tensordict: TensorDict):
        return self.policy_Att_hover.train_op(tensordict)

    def __call__(self, tensordict: TensorDict, deterministic: bool = False):
        self.policy_Opp_SecPass(tensordict, deterministic)
        self.policy_Opp_SecPass_hover(tensordict, deterministic)

        self.policy_Opp_Att_goto(tensordict, deterministic)
        self.policy_Opp_Att(tensordict, deterministic)
        self.policy_Opp_Att_hover(tensordict, deterministic)

        self.policy_FirstPass_goto(tensordict, deterministic)
        self.policy_FirstPass(tensordict, deterministic)
        self.policy_FirstPass_hover(tensordict, deterministic)

        self.policy_SecPass_goto(tensordict, deterministic)
        self.policy_SecPass(tensordict, deterministic)
        self.policy_SecPass_hover(tensordict, deterministic)

        self.policy_Att_goto(tensordict, deterministic)
        self.policy_Att(tensordict, deterministic)
        self.policy_Att_hover(tensordict, deterministic)

        return tensordict