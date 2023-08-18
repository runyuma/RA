from environments.rl_environment import TestEnv,PickOrPlaceEnvWithoutLangReward
import numpy as np
import matplotlib.pyplot as plt
from environments.utils import mouse_demo
from tasks.task import PutBlockInBowl
from stable_baselines3.common.env_checker import check_env
from reward.detector import VILD
import cv2
import clip
from stable_baselines3 import SAC,PPO,TD3
from agents.PolicyNet import CustomCombinedExtractor
from stable_baselines3.common.callbacks import CheckpointCallback
env = PickOrPlaceEnvWithoutLangReward(PutBlockInBowl)
print(env.action_space)
print(env.observation_space)
policy_kwargs = dict(
    features_extractor_class=CustomCombinedExtractor,
    features_extractor_kwargs=dict(features_dim=128),
)
# check_env(env)
# model = SAC("MlpPolicy", env, verbose=1)
policy_name = "sac"
checkpoint_callback = CheckpointCallback(
  save_freq=100,
  save_path="tmp/"+policy_name+"sac_models/",
  name_prefix="rl_model",
  save_replay_buffer=True,
  save_vecnormalize=True,
)
if policy_name == "sac":
    model = SAC("MultiInputPolicy", 
                env, 
                policy_kwargs=policy_kwargs, 
                verbose=1,
                batch_size=16,
                buffer_size=1000,
                train_freq=(1,"episode"),
                tensorboard_log="tmp/sac_tb/",
                )
elif policy_name == "ppo":
    model = PPO("MultiInputPolicy",
                env,
                
                policy_kwargs=policy_kwargs,
                verbose=1,
                tensorboard_log="tmp/ppo_tb/",
                )
model.learn(total_timesteps=2000,log_interval=4,callback=checkpoint_callback)


