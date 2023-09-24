# from environments.rl_environment import TestEnv,PickOrPlaceEnvWithoutLangReward,SimplifyPickOrPlaceEnvWithoutLangReward,SimplifyPickEnvWithoutLangReward
from environments.pickplace_environment import SimplifyPickEnvWithoutLangReward
from environments.dummy_environment import dummypick
import numpy as np
import matplotlib.pyplot as plt
from environments.utils import mouse_demo
from tasks.task import PutBlockInBowl
import cv2
import sys
# env = PickOrPlaceEnvWithoutLangReward(PutBlockInBowl)
env = SimplifyPickEnvWithoutLangReward(render=True,multi_discrete=True)
# env = dummypick(render=True,multi_discrete=True)

np.random.seed(1)


obs,_ = env.reset()
fig, axs = plt.subplots(1, 2, figsize=(13, 7))
plt.cla()
while True:
    # observation = env.super_get_observation()
    # print(observation["image"].shape)
    # image = observation["image"]
    image = env.get_image()
    # print("image:",image.dtype,image.min(),image.max(),image.shape)
    # print(obs['object_in_hand'])
    primitive = int(sys.stdin.readline().strip())
    expert = mouse_demo(image,image)
    pick_point = expert.get_pick_point()
    
    act= np.array([primitive,pick_point[0],pick_point[1]])
    obs, reward, done,_, info = env.step(act)
    # axs[0].imshow(obs["image"][:,:,:3])
    # axs[1].imshow(obs["image"][:,:,3])
    print("#################goal:",obs["lang_goal"])
    print("reward:",reward)
    print("info:",info)
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
