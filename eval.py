from environments.pickplace_environment import SimplifyPickEnvWithoutLangReward,SimplifyPickOrPlaceEnvWithoutLangReward
from environments.pickplace_environment_residual import ResPickOrPlaceEnvWithoutLangReward,ResSimplifyPickOrPlaceEnvWithoutLangReward
import numpy as np
import matplotlib.pyplot as plt
from environments.utils import mouse_demo
from tasks.task import PutBlockInBowl,PickBowl,PutBowlInBowl
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import SAC,PPO,TD3
from agents.LLMRL import LLMSAC,GuideSAC
from agents.PolicyNet import CustomCombinedExtractor
from stable_baselines3.common.callbacks import CheckpointCallback
# env = SimplifyPickOrPlaceEnvWithoutLangReward(render=True)
# env = SimplifyPickEnvWithoutLangReward(render=True,multi_discrete=False)

# env = SimplifyPickOrPlaceEnvWithoutLangReward(render=True,multi_discrete=False,neglect_steps=False)
# model = LLMSAC.load("tmp/explore_sac_models/rl_model_50000_steps.zip")

# env = ResSimplifyPickOrPlaceEnvWithoutLangReward(task= PickBowl,render=True,multi_discrete=False,scale_action=True,scale_obs=True,one_hot_action = True)
# model = LLMSAC.load("tmp/llmsac_res_pickbowl_models/rl_model_40000_steps.zip")

env = ResSimplifyPickOrPlaceEnvWithoutLangReward(task= PutBowlInBowl,render=True,multi_discrete=False,scale_action=True,scale_obs=True,one_hot_action = True)
model =LLMSAC.load("tmp/llmsac_res_putbowl_models/rl_model_50000_steps.zip")
import torch as th
import cv2

# model = SAC.load("tmp/sac_models/rl_model_25000_steps.zip")



episode = 10
episode_num = 0
while episode_num<episode:
    obs,_ = env.reset()
    done = False
    while not done:
        action = model.predict(obs,deterministic=True)[0]
        # action = env.get_expert_demonstration()
        print("action:",action)
        obs,reward,done,_,_ = env.step(action)
    episode_num += 1


# obs = model.policy.obs_to_tensor(obs)[0]
# print("obs",obs)
# pick_map = np.zeros((224,224))
# for i in range(224):
#     for j in range(224):
#         pick_map[i,j] = model.critic(obs,th.tensor([[i,j,0]]))[0][0][0]
# pick_map = pick_map - np.min(pick_map)
# heatmapshow = cv2.applyColorMap(pick_map, cv2.COLORMAP_JET)

# cv2.imshow("pick_map",pick_map)
# image = env.render_image_top()[0]
# cv2.imshow("image",image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


