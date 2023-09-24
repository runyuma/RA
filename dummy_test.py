from environments.rl_environment import TestEnv,PickOrPlaceEnvWithoutLangReward,SimplifyPickOrPlaceEnvWithoutLangReward,SimplifyPickEnvWithoutLangReward
from stable_baselines3 import SAC,PPO,TD3,DDPG,A2C,ppo
from agents.PolicyNet import CustomCombinedExtractor
from stable_baselines3.common.callbacks import CheckpointCallback
import gymnasium as gym
from environments.dummy_environment import dummypick
# import gym
import torch as th
# env = SimplifyPickEnvWithoutLangReward(render=True,multi_discrete=True)
env = dummypick(render=True,multi_discrete=False,scale_obs=True,scale_act=True,obj_num=3)
policy_name = "sac"
# policy_kwargs = dict(activation_fn=th.nn.ReLU,
                    #  net_arch=[256,256,256,256,256,256])
model= SAC(
    "MultiInputPolicy",
    env,
    tensorboard_log="tmp/dummy_tb/",
    # policy_kwargs=policy_kwargs,
)
model.learn(total_timesteps=30000,tb_log_name= "sac 3obj dummy env")