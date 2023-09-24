from environments.rl_environment import TestEnv,PickOrPlaceEnvWithoutLangReward,SimplifyPickOrPlaceEnvWithoutLangReward
import numpy as np
import matplotlib.pyplot as plt
from environments.utils import mouse_demo
from tasks.task import PutBlockInBowl
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import SAC,PPO,TD3
from agents.LLMRL import LLMSAC,GuideSAC
from agents.PolicyNet import CustomCombinedExtractor
from stable_baselines3.common.callbacks import CheckpointCallback
env = SimplifyPickOrPlaceEnvWithoutLangReward(render=True)
import torch as th
import cv2

model = GuideSAC.load("tmp/guide_sac_models/rl_model_40000_steps.zip")

obs,_ = env.reset()
obs = model.policy.obs_to_tensor(obs)[0]
print("obs",obs)
pick_map = np.zeros((224,224))
for i in range(224):
    for j in range(224):
        pick_map[i,j] = model.critic(obs,th.tensor([[i,j,0]]))[0][0][0]
pick_map = pick_map - np.min(pick_map)
heatmapshow = cv2.applyColorMap(pick_map, cv2.COLORMAP_JET)

cv2.imshow("pick_map",pick_map)
image = env.render_image_top()[0]
cv2.imshow("image",image)
cv2.waitKey(0)
cv2.destroyAllWindows()


