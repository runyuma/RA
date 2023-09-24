import torch
import torch.nn as nn
import clip
from torch.nn import functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import numpy as np
import gym
from PIL import Image


class CLIP():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # clip_model, clip_preprocess = clip.load("ViT-B/32",device=device)
    # clip_model.cuda().eval()

class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict,features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules

        device = "cuda" 
        self.device = device

        self.clip_model,self.clip_preprocess = clip.load("ViT-B/32",device=device)
        self.clip_model.eval()
        
        self.Resnet_pick = ResNet18()
        # self.Resnet_place = ResNet18()

        # self.fc1 =  nn.Linear(1539, 512)
        self.fc1 =  nn.Linear(1536, 512)
        self.fc2 =  nn.Linear(512, 256)
        self.fc3 =  nn.Linear(256, 128)



    def forward(self,observation)-> torch.Tensor:
        print("**************forward**************")
        img = observation['image']
        obj_inhand = observation['object_in_hand']
        lang_goal = observation['lang_goal']
        clip_image = observation['clip_image']
        # llm_recommendation = observation['lang_recommendation']
        # if type(llm_recommendation) == torch.Tensor:
        #     lang_recommendation = llm_recommendation.cuda()
        # elif type(llm_recommendation) == np.ndarray:
        #     lang_recommendation = torch.from_numpy(observation['lang_recommendation']).cuda()
        # print(img.shape,img.dtype,img.device)
        # print(obj_inhand.shape,obj_inhand.dtype,obj_inhand.device)

        # clip
        # with torch.no_grad():
        #   # print(lang_goal.shape,lang_goal.dtype,lang_goal.device)
        #   if self.device == "cuda":
        #     text_embeddings =  self.clip_model.encode_text(lang_goal.long())
        #     # clip_image = img[...,0:3]
        #     # print("clip image************",self.clip_preprocess(clip_image).shape)
        #     # clip_image = self.clip_preprocess(Image.fromarray(img)).unsqueeze(0).to(self.device)
        #     # image_features = self.clip_model.encode_image(img).cuda()
        #   else:
        #     text_embeddings =  self.clip_model.encode_text(lang_goal.long()).cuda()
        text_embeddings =  lang_goal.cuda()
            
          # print(text_embeddings.shape,text_embeddings.dtype,text_embeddings.device)
        # resnet
        if len(img.shape) == 3:
            img = img.unsqueeze(0).permute(0,3,1,2)
        elif len(img.shape) == 4:
            img = img.permute(0,3,1,2).cuda()
        # pick = self.Resnet_pick(img)*(1-obj_inhand)
        # place = self.Resnet_place(img)*(obj_inhand)
        pick_place = self.Resnet_pick(img)
        pick = pick_place[:,0:256]*(1-obj_inhand)
        place = pick_place[:,256:]*(obj_inhand)
        # concat
        out = torch.cat((clip_image,text_embeddings,pick,place),dim=1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        return out

 



if __name__ == "__main__":
    from Resnet import ResNet18
    observation = {}
    observation['image'] = torch.zeros((224,224,4))
    observation['object_in_hand'] = 1
    observation['lang_goal'] = clip.tokenize("pick object")
    observation['lang_recommendation'] = np.array([1,2,3])
    observation_space = gym.spaces.Dict({
            "image": gym.spaces.Box(0, 1, (224,224,4,), dtype=np.float32),
            "lang_goal": gym.spaces.Box(0, 49407, (77,), dtype=np.int64),
            "object_in_hand": gym.spaces.Box(0, 1, (1,), dtype=np.bool8),
            "lang_recommendation": gym.spaces.Box(low=np.array([0, 0, 0]),high=np.array([1, 224, 224]),dtype=np.float32),
        })
    policy = CustomCombinedExtractor(observation_space=observation_space)
    out = policy(observation)
else:
    from agents.Resnet import ResNet18
