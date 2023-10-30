import torch
import torch.nn as nn
# import clip
from torch.nn import functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import numpy as np
from gymnasium import spaces
from PIL import Image

# from torchvision.models import resnet18
from .Resnet import ResNet
from .Resnet_utils import BasicBlock
from .attention import AttentionLayer


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict,features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        planes=[16, 32, 64, 128]
        self.image_extractor = ResNet(BasicBlock, 4, [2, 2, 2, 2], planes=planes)
        # print(self.image_extractor)
        input_dim = planes[-1] + observation_space['image'].shape[0] + observation_space['object_in_hand'].shape[0] + observation_space['lang_goal'].shape[0]
        self.bn1 = nn.BatchNorm1d(planes[-1])
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, features_dim)



    def forward(self,observation)-> torch.Tensor:
        print("**************forward**************")
        pos = observation['image']
        obj_inhand = observation['object_in_hand']
        lang_goal = observation['lang_goal']
        image = observation['rgbd']
 
        if len(image.shape) == 3:
            image = image.unsqueeze(0).permute(0,3,1,2)
        elif len(image.shape) == 4:
            image = image.permute(0,3,1,2).cuda()
        image = self.image_extractor(image)
        image = self.bn1(image)
        
        out = torch.cat((image,pos,obj_inhand,lang_goal),dim=1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        return out
    
class FeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict,features_dim: int = 32):
        super().__init__(observation_space, features_dim)
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        planes=[16, 32, 64, 128]
        self.image_extractor = ResNet(BasicBlock, 4, [2, 2, 2, 2], planes=planes)
        # print(self.image_extractor)
        input_dim = planes[-1] + observation_space['image'].shape[0] + observation_space['object_in_hand'].shape[0] + observation_space['lang_goal'].shape[0]
        print("input_dim:",input_dim,observation_space['image'].shape[0])
        self.bn1 = nn.BatchNorm1d(planes[-1])
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, features_dim)
        



    def forward(self,observation)-> torch.Tensor:
        print("**************forward**************")
        pos = observation['image']
        obj_inhand = observation['object_in_hand']
        lang_goal = observation['lang_goal']
        image = observation['rgbd']
        # print("pos:",pos)
 
        if len(image.shape) == 3:
            image = image.unsqueeze(0).permute(0,3,1,2)
        elif len(image.shape) == 4:
            image = image.permute(0,3,1,2).cuda()
        
        image = self.image_extractor(image)
        image = self.bn1(image)
        
        out = torch.cat((image,pos,obj_inhand,lang_goal),dim=1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        return out
    
class AttentionFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict,features_dim: int = 32, extract_size = 12):
        super().__init__(observation_space, features_dim)
        self.layer1 = nn.Sequential(
            nn.Conv2d(4, 6, kernel_size=5, stride=2, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, extract_size, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(extract_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc1 = nn.Sequential(
            nn.Linear(extract_size, extract_size),
            nn.ReLU())
        self.token_size = extract_size+2+observation_space['object_in_hand'].shape[0]+observation_space['lang_goal'].shape[0]+1
        self.atten = AttentionLayer(self.token_size, 1)
        self.fc2 = nn.Sequential(
            nn.Linear(self.token_size, self.token_size),
            nn.ReLU())
        self.fc3 = nn.Sequential(
            nn.Linear(self.token_size, features_dim),
            nn.ReLU())

        
    def forward(self,observation)-> torch.Tensor:
        pos = observation['image']
        obj_inhand = observation['object_in_hand']
        lang_goal = observation['lang_goal']
        image = observation['rgbd']

        if len(image.shape) == 4:
            # Add a dummy batch dimension
            image = image.unsqueeze(0)
            obj_inhand = obj_inhand.unsqueeze(0)
            lang_goal = lang_goal.unsqueeze(0)
            pos = pos.unsqueeze(0)
        
        image = image.permute(0, 1, 4, 2, 3)
        B, N, C, H, W = image.shape
        image = image.reshape(B * N, C, H, W)
        image = self.layer1(image)
        image = self.layer2(image)
        image = image.reshape(B * N, -1)
        image = self.fc1(image)
        image = image.reshape(B, N, -1)

        pos = pos.reshape(B, N, -1)
        out = torch.cat((image,pos),dim=2)

        obj_inhand = obj_inhand.unsqueeze(1).repeat_interleave(N, dim=1)
        lang_goal = lang_goal.unsqueeze(1).repeat_interleave(N, dim=1)
        position = torch.arange(N).unsqueeze(0).repeat_interleave(B, dim=0).unsqueeze(2)
        out = torch.cat((out,obj_inhand,lang_goal,position),dim=2)
        out = self.atten(out,out,out)
        out = out.reshape(B,-1)
        out = self.fc2(out)
        out = self.fc3(out)

        return out


 





