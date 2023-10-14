from environments.pickplace_environment import SimplifyPickEnvWithoutLangReward,SimplifyPickOrPlaceEnvWithoutLangReward
from agents.LLMRL import LLMSAC,GuideSAC
import numpy as np
import matplotlib.pyplot as plt
from environments.utils import mouse_demo
from tasks.task import PutBlockInBowl
from stable_baselines3.common.env_checker import check_env
from reward.detector import VILD
import torch as th
import cv2
# import clip
from stable_baselines3 import SAC,PPO,TD3,DDPG,A2C,ppo
from agents.PolicyNet import CustomCombinedExtractor
from stable_baselines3.common.callbacks import CheckpointCallback
import gymnasium as gym
from environments.dummy_environment import dummypick
# import gym
# env = SimplifyPickEnvWithoutLangReward(render=True,multi_discrete=True)
# env = dummypick(render=True,multi_discrete=False,scale_obs=True,obj_num=3)
# env = SimplifyPickEnvWithoutLangReward(render=True,multi_discrete=False,scale_obs=True)
env = SimplifyPickOrPlaceEnvWithoutLangReward(render=True,multi_discrete=False)
# policy_name = "sac"
policy_name = "explore_sac"
checkpoint_callback = CheckpointCallback(
  save_freq=5000,
  save_path="tmp/"+policy_name+"_models/",
  name_prefix="rl_model",
  save_replay_buffer=True,
  save_vecnormalize=True,
)

model= LLMSAC(
    "MultiInputPolicy",
    env,
    learning_rate=0.0006,
    gamma=0.9,
    epsilon_llm=0.15,
    tensorboard_log="tmp/pickplace_tb/",
)
model.learn(total_timesteps=50000,callback=checkpoint_callback,tb_log_name= "llmsac_ep0.15")