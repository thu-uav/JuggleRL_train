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


from .mappo import MAPPOPolicy
from .ppo import *
# from .test_single import Policy
# from .mappo_formation import PPOFormation as Policy
from ._ppo import PPOPolicy as Policy
from .happo import HAPPOPolicy
from .qmix import QMIXPolicy

from .dqn import DQNPolicy
from .sac import SACPolicy
from .td3 import TD3Policy
from .matd3 import MATD3Policy
from .tdmpc import TDMPCPolicy
from .ppo_sp import PPOSPPolicy
from .nfsp import NFSPPolicy
from .psro import PSROPolicy,PSROSPPolicy

from .mappo_mask import MAPPOPolicy_mask
from .volleyball import (
    MAPPOPolicy_SecPass_Att, 
    MAPPOPolicy_SecPass_Att_w_hover, 
    MAPPOPolicy_Att,
    MAPPOPolicy_Att_hover,
    MAPPOPolicy_Block_multi, 
    MAPPOPolicy_SecPass_w_hover,
    MAPPOPolicy_FirstPass_multi,
    MAPPOPolicy_FirstPass_multi_hover,
    MAPPOPolicy_SecPass_multi,
    MAPPOPolicy_SecPass_multi_hover,
    MAPPOPolicy_Att_multi,
    MAPPOPolicy_Att_multi_hover,
    MAPPOPolicy_full_game,
)

