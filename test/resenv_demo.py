# from environments.rl_environment import TestEnv,PickOrPlaceEnvWithoutLangReward,SimplifyPickOrPlaceEnvWithoutLangReward,SimplifyPickEnvWithoutLangReward
import os
import sys
# print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environments.pickplace_environment_residual import ResPickOrPlaceEnvWithoutLangReward,ResSimplifyPickOrPlaceEnvWithoutLangReward
from environments.dummy_environment import dummypick
from tasks.task import PutBowlInBowl,PickBowl
import numpy as np
import matplotlib.pyplot as plt
from environments.utils import mouse_demo
from tasks.task import PutBlockInBowl
import cv2
import sys

one_hot = True
residual = True
# env = ResSimplifyPickOrPlaceEnvWithoutLangReward(render=True,multi_discrete=True,scale_action=True,ee = "suction",scale_obs=True,one_hot_action = True)
env = ResPickOrPlaceEnvWithoutLangReward(image_obs=True,residual=residual,observation_noise=5,render=True,multi_discrete=False,scale_action=True,ee ="suction",scale_obs=True,one_hot_action = True)
# env = ResSimplifyPickOrPlaceEnvWithoutLangReward(task= PickBowl,render=True,multi_discrete=True,scale_action=True,scale_obs=True,one_hot_action = True)
# env = ResSimplifyPickOrPlaceEnvWithoutLangReward(task= PutBowlInBowl,render=True,multi_discrete=True,scale_action=True,scale_obs=True,ee = "suction",one_hot_action = True)
# env = ResPickOrPlaceEnvWithoutLangReward(render=True,multi_discrete=True,scale_action=True,scale_obs=True,one_hot_action = True)
print(env.observation_space)
print(env.action_space)

np.random.seed(0)
obs,_ = env.reset()
fig, axs = plt.subplots(1, 2, figsize=(13, 7))
plt.cla()
while True:
    if env.image_obs:
        print(obs["rgbd"][:,:,:3].max())
        # cv2.imshow("color0",obs["rgbd"][:,:,:3])
        # cv2.imshow("depth0",obs["rgbd"][:,:,3])
        image = env.get_image()[:, :, ::-1]
        # cv2.imshow("color0",image)
        # cv2.imshow("depth0",20*env.get_depth())
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    image = env.get_image()
    print("obs: ", obs["image"])
    print("primitive: 0 for pick, 1 for place")
    primitive = int(sys.stdin.readline().strip())
    if primitive != 2:
        print("obj_num: 0 for 1 obj, 1 for 2 obj, 2 for 3 obj")
        print(env.task.category_names)
        if residual:
            obj_num = int(sys.stdin.readline().strip())
            if one_hot:
                obj_one_hot = np.zeros(len(env.task.category_names))
                obj_one_hot[obj_num] = 1
        expert = mouse_demo(image,image)
        pick_point = expert.get_pick_point()
        print("pick_point:",pick_point)
        if residual:
            obs_pos = obs["image"][2*obj_num:2*obj_num+2]
            if env.scale_obs:
                obs_pos = obs_pos*224
            res = np.array(pick_point) - obs_pos
            if env.scale_action:
                res = res/112
            if one_hot:
                act = np.concatenate((np.array([primitive]),obj_one_hot,res))
            else:
                act = np.concatenate((np.array([primitive,obj_num]),res))
        else:
            act = np.concatenate((np.array([primitive]),pick_point))
    else:
        act = env.get_expert_demonstration()
        print("expert demonstration:",act)
    obs, reward, done,_, info = env.step(act)
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
