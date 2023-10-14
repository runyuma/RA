from environments.rl_environment import TestEnv,PickOrPlaceEnvWithoutLangReward,SimplifyPickOrPlaceEnvWithoutLangReward,SimplifyPickEnvWithoutLangReward
import numpy as np
import matplotlib.pyplot as plt
from environments.utils import mouse_demo
from tasks.task import PutBlockInBowl
from stable_baselines3.common.env_checker import check_env
from reward.detector import VILD
import torch as th
import cv2
import clip
from stable_baselines3 import SAC,PPO,TD3,DDPG,A2C,ppo
from agents.PolicyNet import CustomCombinedExtractor
from stable_baselines3.common.callbacks import CheckpointCallback
import gymnasium as gym
from environments.dummy_environment import dummypick
# import gym
# env = SimplifyPickEnvWithoutLangReward(render=True,multi_discrete=True)
env = dummypick(render=True,multi_discrete=False,scale_obs=True,obj_num=3)
policy_name = "a2c"
# checkpoint_callback = CheckpointCallback(
#   save_freq=500000,
#   save_path="tmp/"+policy_name+"_models/",
#   name_prefix="rl_model",
#   save_replay_buffer=True,
#   save_vecnormalize=True,
# )
# policy_kwargs = dict(activation_fn=th.nn.ReLU,
#                      net_arch=[256,256,256,256,256,256]
#                     # net_arch=[64,64,64,64]
#                      )

model= SAC(
    "MultiInputPolicy",
    env,
    learning_rate=0.0006,
    gamma=0.9,
    train_freq = 1,
    # gradient_steps=-1,
    tensorboard_log="tmp/a2c_tb/",
)
model.learn(total_timesteps=30000,tb_log_name= "dummy env")