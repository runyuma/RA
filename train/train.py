import os
import sys
# print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environments.pickplace_environment_residual import ResPickOrPlaceEnvWithoutLangReward,ResSimplifyPickOrPlaceEnvWithoutLangReward
from tasks.task import PickBowl,PutBowlInBowl
from agents.LLMRL import LLMSAC,GuideSAC
from agents.PolicyNet import CustomCombinedExtractor,FeatureExtractor
import numpy as np
import matplotlib.pyplot as plt


from stable_baselines3 import SAC,PPO,TD3,DDPG,A2C,ppo
from stable_baselines3.common.callbacks import CheckpointCallback
policy_name = "llmsac_imgdrop_withnoise"
ep = (0.15,"fixed")
name = policy_name + str(100*ep[0])+ep[1]+task_name+"_model"
checkpoint_callback = CheckpointCallback(
save_freq=5000,
save_path="tmp/"+ name +"/",
name_prefix="rl_model",
save_replay_buffer=False,
save_vecnormalize=True,
)
residual = True
env = ResPickOrPlaceEnvWithoutLangReward(
image_obs=True,
residual=residual,
observation_noise=5,
render=False,
multi_discrete=False,
scale_action=True,
ee ="suction",
scale_obs=True,
one_hot_action = True)