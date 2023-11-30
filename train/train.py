import os
import sys
# print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environments.pickplace_environment_residual import ResPickOrPlaceEnvWithoutLangReward
from tasks.task import PickBowl,PutBowlInBowl
from agents.LLMRL import LLMSAC
from agents.PolicyNet import CustomCombinedExtractor,FeatureExtractor,AttentionFeatureExtractor
import numpy as np
import matplotlib.pyplot as plt
from tasks.letter import PutLetterOontheBowl,PutLetterontheBowl

from stable_baselines3 import SAC,PPO,TD3,DDPG,A2C,ppo
from stable_baselines3.common.callbacks import CheckpointCallback
def train():
    render = False
    args = sys.argv
    if len(args) > 1:
        if args[1] == "render":
            render = True
    policy_name = "llmsac_imgatten_withnoise"
    task_name = "PutLetterontheBowl"
    ep = (0.15,"fixed")
    # ep = (1,"fixed")
    name = policy_name + str(100*ep[0])+ep[1]+task_name+"_model"
    checkpoint_callback = CheckpointCallback(
    save_freq=5000,
    save_path="tmp/"+ name +"/",
    name_prefix="rl_model",
    save_replay_buffer=False,
    save_vecnormalize=True,
    )

    env = ResPickOrPlaceEnvWithoutLangReward(
        task= PutLetterontheBowl,
        image_obs=True,
        residual=True,
        observation_noise=5,
        render=render,
        multi_discrete=False,
        scale_action=True,
        ee="suction",
        scale_obs=True,
        neglect_steps=True,
        one_hot_action = True)
    policy_kwargs = dict(
        features_extractor_class=AttentionFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=32),
    )
    print(env.observation_space)
    print(env.action_space)

    model= LLMSAC(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_starts=50,
        learning_rate=0.5*3e-4,
        buffer_size= 80000,
        gamma=0.9,
        epsilon_llm=ep,
        tensorboard_log="tmp/pickplace_tb/",
    )

    model.learn(total_timesteps=80000,callback=checkpoint_callback,tb_log_name= name)
if __name__ == "__main__":
    train()