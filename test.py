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
# env = dummypick(render=True,multi_discrete=False,scale_obs=True,obj_num=3)
env = SimplifyPickEnvWithoutLangReward(render=True,multi_discrete=False,scale_obs=True)
policy_name = "sac"
checkpoint_callback = CheckpointCallback(
  save_freq=5000,
  save_path="tmp/"+policy_name+"_models/",
  name_prefix="rl_model",
  save_replay_buffer=True,
  save_vecnormalize=True,
)

model= SAC(
    "MultiInputPolicy",
    env,
    tensorboard_log="tmp/dummy_tb/",
)
model.learn(total_timesteps=30000,callback=checkpoint_callback,tb_log_name= "sac 3obj pick env")