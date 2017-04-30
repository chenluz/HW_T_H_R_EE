import gym
import copy
import deeprl_hw3
import numpy as np
from deeprl_hw3.ilqr import calc_ilqr_input
env_name = 'TwoLinkArm-v0'; # Change for different env
env = gym.make(env_name);
env_copy = copy.deepcopy(env);
tN = 100;
u_array = calc_ilqr_input(env, env_copy, tN, max_iter=10000, is_warm_start = False)

