from environments.rl_environment import TestEnv,PickOrPlaceEnvWithoutLangReward
import numpy as np
import matplotlib.pyplot as plt
from environments.utils import mouse_demo
from tasks.task import PutBlockInBowl
from reward.detector import VILD
import cv2
import clip
env = PickOrPlaceEnvWithoutLangReward(PutBlockInBowl)
np.random.seed(1)


category_names = ['yellow block', 'green block', 'blue block', 'yellow bowl', 'green bowl', 'blue bowl']
obs = env.reset()
fig, axs = plt.subplots(1, 2, figsize=(13, 7))
plt.cla()
while True:
    # print("obs:",obs["image"].dtype)
    # print(obs['object_in_hand'])
    expert = mouse_demo(obs["image"][:,:,:3],obs["image"][:,:,3])
    pick_point = expert.get_pick_point()

    act= np.array([obs['object_in_hand'][0],pick_point[0]/224,pick_point[1]/224])
    obs, reward, done, info = env.step(act)
    axs[0].imshow(obs["image"][:,:,:3])
    axs[1].imshow(obs["image"][:,:,3])
    plt.pause(1)
    
    
    if done:
        break
# plt.subplot(1, 3, 1)
# img = obs["image"]
# plt.title('color-view')
# plt.imshow(img)
# plt.subplot(1, 3, 2)
# img = env.get_camera_image_top()
# img = obs["depth"]
# plt.title('depth-view')
# plt.imshow(img)
# plt.subplot(1,3,3)
# plt.imshow(env.get_camera_image())
# plt.show()
