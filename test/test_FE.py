# template for test file
import os
import sys
# print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
# import clip
from torch.nn import functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import numpy as np
from gymnasium import spaces
from PIL import Image

# from torchvision.models import resnet18
from agents.Resnet import ResNet
from agents.Resnet_utils import BasicBlock
from agents.PolicyNet import CustomCombinedExtractor

if __name__ == "__main__":
    # from Resnet import ResNet18
    image_size = (64,64)
    obj_num = 3
    goal_num = 2

    observation = {}
    observation['rgbd'] = torch.zeros((64,64,4))
    observation['object_in_hand'] = torch.tensor([[1]])
    observation['lang_goal'] = torch.zeros((1,goal_num,))
    observation['image'] = torch.zeros((1,obj_num*2,))

    observation_space = spaces.Dict({
                "rgbd": spaces.Box(0, 1, image_size + (4,), dtype=np.float32),
                "image": spaces.Box(0, 1, (obj_num*2,), dtype=np.float32),
                "lang_goal": spaces.Box(0, obj_num, (goal_num,), dtype=np.int32),
                "object_in_hand": spaces.Box(0, 1, (1,), dtype=np.bool8),
            })
    policy = CustomCombinedExtractor(observation_space=observation_space)
    out = policy(observation)