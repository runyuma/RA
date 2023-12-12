import os
import sys
# print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environments.pickplace_environment_residual import ResPickOrPlaceEnvWithoutLangReward
import numpy as np

# import matplotlib.pyplot as plt
from environments.utils import mouse_demo
from tasks.letter import PutLetterOontheBowl,PutLetterontheBowl,PutblockonBowlSameColor
import cv2
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
task = PutLetterontheBowl
task = PutblockonBowlSameColor
env = ResPickOrPlaceEnvWithoutLangReward(
                                        task= task,
                                         image_obs=True,
                                         residual=True,
                                         observation_noise=5,
                                         render=True,
                                         multi_discrete=False,
                                         scale_action=True,
                                         ee="suction",
                                         scale_obs=True,
                                         neglect_steps=False,
                                         one_hot_action = True)
residual = True
one_hot = True
np.random.seed(1)
obs,_ = env.reset()
plot_num = env.observation_space["rgbd"].shape[0]
# print("obj_num:",plot_num,env.observation_space["rgbd"])

while True:
    plt.clf()
    if env.image_obs:
        for i in range(plot_num):
            
            plt.subplot(2,plot_num,i+1)
            plt.imshow(obs["rgbd"][i,:,:,:3])
            plt.subplot(2,plot_num,i+1+plot_num)
            plt.imshow(obs["rgbd"][i,:,:,3])
        plt.show()
            
        # print(obs["rgbd"].shape)
        # # print(obs["rgbd"][:,:,:3].max())
        # # cv2.imshow("color0",obs["rgbd"][:,:,:3])
        # # cv2.imshow("depth0",obs["rgbd"][:,:,3])
        # image = env.get_image()[:, :, ::-1]
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
                res = res/env.residual_region
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
    if done:
        obs,_ = env.reset()
    print("#################goal:",obs["lang_goal"])
    print("reward:",reward)
    print("info:",info)
    plt.pause(1)