import os
import sys
# print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environments.pickplace_environment_residual import ResPickOrPlaceEnvWithoutLangReward
from tasks.task import PickBowl,PutBowlInBowl
from stable_baselines3.common.utils import set_random_seed
from agents.LLMRL import LLMSAC
from agents.PolicyNet import CustomCombinedExtractor,FeatureExtractor,AttentionFeatureExtractor
import numpy as np
import torch
import matplotlib.pyplot as plt
from tasks.letter import PutLetterOontheBowl,PutLetterontheBowl,PutblockonBowlSameColor
import argparse
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

from stable_baselines3 import SAC,PPO,TD3,DDPG,A2C,ppo
from stable_baselines3.common.callbacks import CheckpointCallback
def train():
    parser = argparse.ArgumentParser(description='Training script.')

    # Add arguments
    parser.add_argument('--seed', type=int, default=4, help='Random seed.')
    parser.add_argument('--iters', type=int, default=80000, help='Number of iterations.')
    parser.add_argument('--save', type=str2bool, default=True, help='Whether to save the model.')
    parser.add_argument('--render', type=str2bool, default=False, help='Whether to render the environment.')
    parser.add_argument('--save_freq', type=int, default=2500, help='save frequency')
    parser.add_argument('--device', type=int, default=0, help='device')

    # Parse arguments
    args = parser.parse_args()
    seed = args.seed
    set_random_seed(seed,using_cuda=True)
    render = args.render
    save = args.save
    iters = args.iters
    save_freq = args.save_freq
    device = args.device
    if device == 0:
        device = "cuda"
    else:
        device = "cpu"
    print("render: ",render)



    policy_name = "llmsac_imgatten_withnoise"
    task_name = "PutLetterontheBowl"
    # task_name = "PutblockonBowlSameColor"
    ep = (0.15,"fixed")
    # ep = (1,"fixed")
    name =policy_name+"_"+task_name+"seed"+str(seed)+"_model"
    if save:
        checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path="tmp/"+ name +"/",
        name_prefix="rl_model",
        save_replay_buffer=False,
        save_vecnormalize=True,
        )

    env = ResPickOrPlaceEnvWithoutLangReward(
        task= PutLetterontheBowl,
        # task= PutblockonBowlSameColor,
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

    if save:
        model= LLMSAC(
            "MultiInputPolicy",
            env,
            policy_kwargs=policy_kwargs,
            learning_starts= 100,
            learning_rate=0.5*3e-4,
            buffer_size= iters,
            gamma=0.9,
            epsilon_llm=ep,
            tensorboard_log="tmp/final_tb/",
            device=device
        )
    else:
        model= LLMSAC(
            "MultiInputPolicy",
            env,
            policy_kwargs=policy_kwargs,
            learning_starts= 100,
            learning_rate=0.5*3e-4,
            buffer_size= iters,
            gamma=0.9,
            epsilon_llm=ep,
        )

    model.learn(total_timesteps=iters,callback=checkpoint_callback,tb_log_name= name)
if __name__ == "__main__":
    train()