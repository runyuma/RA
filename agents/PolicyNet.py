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
        
        
        # batch_size,_,row_size,col_size = image.shape
        # row = torch.arange(0, row_size).unsqueeze(0).expand(col_size, -1)/row_size
        # col = torch.arange(0, col_size).unsqueeze(1).expand(-1, row_size)/col_size
        # tensor = torch.stack((row, col), dim=0).unsqueeze(0).expand(batch_size, -1, -1, -1).cuda()
        # image = torch.cat((image,tensor),dim=1)
        # image = image[:,3:,:,:]
        image = self.image_extractor(image)
        image = self.bn1(image)
        
        out = torch.cat((image,pos,obj_inhand,lang_goal),dim=1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        return out

 





