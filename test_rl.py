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
from stable_baselines3 import SAC,PPO,TD3,DDPG,A2C
from agents.LLMRL import LLMSAC,GuideSAC
from agents.PolicyNet import CustomCombinedExtractor
from agents.PrioritizedReplay import PrioritizedReplayBuffer
from stable_baselines3.common.callbacks import CheckpointCallback
import gymnasium as gym
# import gym
env = SimplifyPickEnvWithoutLangReward(render=True)
# env = gym.make("SimplePickAndPlace-v0")
# policy_name = "simp_ddpg"
# policy_name = "simp_sac"
policy_name = "guide_sac"
# policy_name = "sac"
checkpoint_callback = CheckpointCallback(
  save_freq=10000,
  save_path="tmp/"+policy_name+"_models/",
  name_prefix="rl_model",
  save_replay_buffer=True,
  save_vecnormalize=True,
)
step = int(200000)
if policy_name == "simp_sac":
    policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=[256,256,256,256])
    model = LLMSAC("MultiInputPolicy", 
                env, 
                verbose=1,
                learning_rate= 0.001,
                tau=0.01,
                learning_starts=200,
                batch_size=128,
                gradient_steps=-1,
                buffer_size=step,
                epsilon_llm=0.0,
                epsilon_decrease=0.0000,
                train_freq=(5,"step"),
                # ent_coef=1,
                # tensorboard_log="tmp/sim_sac_tb/",
                policy_kwargs=policy_kwargs,
                device="cuda",
                )
if policy_name == "guide_sac":
    policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=[64,64,64])
    model = GuideSAC("MultiInputPolicy", 
                env, 
                verbose=1,
                learning_rate= 0.0005,
                tau = 0.15,
                learning_starts=1500,
                batch_size=256,
                gradient_steps=-1,
                replay_buffer_class=PrioritizedReplayBuffer,
                buffer_size=200000,
                train_freq=(5,"step"),
                # ent_coef=1,
                # tensorboard_log="tmp/guide_sac_tb/",
                policy_kwargs=policy_kwargs,
                device="cuda",
                )
    print(env.observation_space)
    print(model.policy)

# # model.load_replay_buffer("tmp/replay_buffer/"+policy_name)
# model.replay_buffer.load_replay_buffer("tmp/replay_buffer/"+policy_name)
# print("replay buffer size:",model.replay_buffer.actions.shape,model.replay_buffer.pos)
model.learn(total_timesteps=100,log_interval=4,callback=checkpoint_callback)
# print("replay buffer size:",model.replay_buffer.actions.shape,model.replay_buffer.pos)
# # print("replay buffer size:",model.replay_buffer.pos)
# model.save_replay_buffer("tmp/replay_buffer/"+policy_name+".pkl")
