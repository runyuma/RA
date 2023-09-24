from typing import Any
import gymnasium as gym
from gym import spaces
import numpy as np
from typing import TYPE_CHECKING, Any, Generic, SupportsFloat, TypeVar
from gymnasium import logger, spaces
from gymnasium.utils import RecordConstructorArgs, seeding
import cv2

class dummypick(gym.Env):
    def __init__(self,render=False,multi_discrete=True,scale_obs = False,scale_act = False,obj_num=3,goal_num=1):
        self.render_mode=render
        self.image_size = (224,224)
        self.block_size = 20
        self.max_steps = 10
        self.obj_num = obj_num
        self.scale_obs = scale_obs
        self.scale_act = scale_act
        if not self.scale_obs:
            self.observation_space = self.observation_space = spaces.Dict({
                "image": spaces.Box(0+self.block_size, self.image_size[0]-self.block_size, (obj_num*2,), dtype=np.int32),
                "lang_goal": spaces.Box(0, obj_num-1, (goal_num,), dtype=np.int32),
                "object_in_hand": spaces.Box(0, 1, (1,), dtype=np.int8),
            })
        else:
            self.observation_space = self.observation_space = spaces.Dict({
                "image": spaces.Box((0+self.block_size)/self.image_size[0], (self.image_size[0]-self.block_size)/self.image_size[0], (obj_num*2,), dtype=np.float32),
                "lang_goal": spaces.Box(0, obj_num-1, (goal_num,), dtype=np.int32),
                "object_in_hand": spaces.Box(0, 1, (1,), dtype=np.int8),
            })
        self.colors = [
            (255,0,0),
            (0,255,0),
            (0,0,255),
        ]
        if not multi_discrete:
            if not scale_act:
                self.action_space = spaces.Box(low=np.array([0, 0, 0]),high=np.array([1, self.image_size[0], self.image_size[1]]),dtype=np.float32)
            else:
                self.action_space = spaces.Box(low=np.array([0, 0, 0]),high=np.array([1, 1, 1]),dtype=np.float32)
        else:
            self.action_space = spaces.MultiDiscrete([2, self.image_size[0], self.image_size[1]])
    
    def reset(
        self,
        seed = None,
        options = None,
        ) :
        super().reset(seed=seed, options=options)
        self.observation_space.seed(seed)
        
        obs = self.observation_space.sample()
        obs["object_in_hand"][0] = 0
        self.observation = obs
        print("observation:",self.observation)

        self.step_count = 0
        self.render()
        return obs,{}
    
    def step(self,action):
        _action = action
        # print("raw action:",_action)
        action = self.get_real_action(action)
        obs,reward,done = self.get_observations_and_reward(action)
        self.render()
        self.observation = obs
        print("observation:",self.observation)
        print("action:",action,_action)
        print("reward:",reward)
        self.step_count += 1
        done |= self.step_count >= self.max_steps
        return obs,reward,done,False,{}
    
    def get_observations_and_reward(self,action):
        threshold = self.block_size/2

        goal_idx = self.observation["lang_goal"][0]
        desired_position = self.get_desired_position()
        # print("desired_pos:",desired_pos)
        # print("action:",action)
        dis = np.linalg.norm(desired_position-np.array(action[1:]))
        if dis < threshold:
            obs = self.observation
            obs["object_in_hand"] = 1
            obs["image"][2*goal_idx:2*goal_idx+2] = [0,0]
            reward = 1
            done = True
        else:
            obs = self.observation
            reward = 0
            done = False
            phase_penalty = 0.25

            desired_primitive = self.observation["object_in_hand"]
            desired_action = [desired_primitive]+desired_position

            if (action[0] > 0.5)== desired_primitive:
                phase_loss = 0
            else:
                phase_loss = - phase_penalty
             
            dis = np.linalg.norm(action[1:]-desired_position)

            thre = 100
            dis_loss = - (dis/thre,1)[dis>thre]**2*0.25
            print("dis_loss:",dis,dis_loss,action[1:],desired_position)
            reward += phase_loss+dis_loss

        return obs,reward,done
    def get_desired_position(self):
        goal_idx = self.observation["lang_goal"][0]
        desired_pos = np.array(self.observation["image"])[2*goal_idx:2*goal_idx+2]
        if self.scale_obs:
            desired_pos = desired_pos*self.image_size[0]
        return desired_pos
    def get_real_action(self,action):
        _action = np.zeros_like(action)
        _action[0] = action[0]
        if self.scale_act:
            _action[1:] = action[1:]*self.image_size[0]
        else:
            _action[1:] = action[1:]
        return _action
    
    def render(self):
        if self.render_mode:
            print("rendering")
            img = np.zeros(self.image_size+(3,),dtype=np.uint8) 
            for i in range(self.obj_num):
                if self.scale_obs:
                    pos = (self.observation["image"][2*i:2*i+2]*self.image_size[0]).astype(np.int32)
                else:
                    pos = self.observation["image"][2*i:2*i+2]
                l = int(self.block_size/2)
                color = self.colors[i]
                cv2.rectangle(img,(pos[1]-l,pos[0]-l),(pos[1]+l,pos[0]+l),color,-1)
            return img
        
if __name__ == "__main__":
    env = dummypick(render=True)
    obs,info = env.reset(seed=0)
    img = env.render()
    cv2.imshow("img",img) 
    cv2.waitKey(0)
    cv2.destroyAllWindows()   
        
