from environments.environment import PickPlaceEnv,BOUNDS,PIXEL_SIZE
from tasks.task import PutBlockInBowl,PickBlock
from environments.timer import limit_decor
import numpy as np
import pybullet
import cv2
from environments.utils import *
# from gym import spaces
from gymnasium import spaces
# import torch as th
import time
from PIL import Image
import time


class ResPickOrPlaceEnvWithoutLangReward(PickPlaceEnv):
    # choose pick/place instead of pick_and_place

    def __init__(self, 
                task= PutBlockInBowl,
                residual = True,
                image_obs = False,
                observation_noise = 0,
                languange_expert = False,
                multi_discrete = False,
                scale_obs = True,
                scale_action = True,
                neglect_steps = True,
                ee = "gripper", # suction
                render=False,
                one_hot_action = False):
        super().__init__(task,render=render,ee=ee)
        self.residual = residual
        image_size = (224,224)
        self.image_size = image_size
        self.reshape_para = 0.25
        self.height_coef = 20
        self.observation_noise = observation_noise

        self.image_obs = image_obs
        self.neglect_steps = neglect_steps
        self.scale_obs = scale_obs
        self.scale_action = scale_action

        obj_num = len(self.task.category_names)
        goal_num = self.task.goal_num
        self.one_hot_action = one_hot_action
        if hasattr(self.task,"residual_region"):
            self.residual_region = self.task.residual_region
        else:
            self.residual_region = self.image_size[0]/2

        if image_obs:
            if not self.residual:
                image_size = (int(image_size[0]*self.reshape_para),int(image_size[1]*self.reshape_para))
                
                self.observation_space = spaces.Dict({
                        "rgbd": spaces.Box(0, 1, image_size + (4,), dtype=np.float32),
                        "image": spaces.Box(0, 1, (obj_num*2,), dtype=np.float32),
                        "lang_goal": spaces.Box(0, obj_num, (goal_num,), dtype=np.int32),
                        "object_in_hand": spaces.Box(0, 1, (1,), dtype=np.bool8),
                    })
            else:
                h = int(self.residual_region*2)
                w = int(self.residual_region*2)
                self.observation_space = spaces.Dict({
                        "rgbd": spaces.Box(0, 1, (obj_num,h,w,4), dtype=np.float32),
                        "image": spaces.Box(0, 1, (obj_num*2,), dtype=np.float32),
                        "lang_goal": spaces.Box(0, obj_num, (goal_num,), dtype=np.int32),
                        "object_in_hand": spaces.Box(0, 1, (1,), dtype=np.bool8),
                    })
        else:
            if not scale_obs:
                self.observation_space = spaces.Dict({
                    "image": spaces.Box(0, self.image_size[0], (obj_num*2,), dtype=np.int32),
                    "lang_goal": spaces.Box(0, obj_num, (goal_num,), dtype=np.int32),
                    "object_in_hand": spaces.Box(0, 1, (1,), dtype=np.bool8),
                })
            else:
                self.observation_space = spaces.Dict({
                    "image": spaces.Box(0, 1, (obj_num*2,), dtype=np.float32),
                    "lang_goal": spaces.Box(0, obj_num, (goal_num,), dtype=np.int32),
                    "object_in_hand": spaces.Box(0, 1, (1,), dtype=np.bool8),
                })


        if self.residual:
            if self.one_hot_action:
                low = - np.ones((obj_num+3,))
                low[0:-2] = 0*low[0:-2]
                high = np.ones((obj_num+3,))
                if not scale_action:
                    high[-2] = self.residual_region
                    high[-1] = self.residual_region
                self.action_space = spaces.Box(low=low,high=high,dtype=np.float32)
            else:
                low = - np.ones((4,))# motion primitive/object index/position
                high = np.ones((4,))
                if not scale_action:
                    high[-2] = self.residual_region
                    high[-1] = self.residual_region
                self.action_space = spaces.Box(low=low,high=high,dtype=np.float32)
        else:
            low = np.zeros((3,))
            high = np.ones((3,))
            if not scale_action:
                high[-2] = self.residual_region
                high[-1] = self.residual_region
            self.action_space = spaces.Box(low=low,high=high,dtype=np.float32)
                # self.action_space = spaces.Box(low=np.array([0, 0, 0]),high=np.array([1, self.image_size[0], self.image_size[1]]),dtype=np.float32)


        self.max_steps = 10 # task.max_steps


    def reset(self,seed = None):
        self.last_pick_success = 0
        self.info = {
            "step_result":['initialization'],
            "LLM_recommendation":[],
            "obj_position":[],
        }
        # get observation && ground truth position
        self.observation = None
        obs,info = super().reset()
        self.observation = obs
        return obs,info
    
    def translate_action(self,action,observation):
        # translate action to real action [primitive,position in pixel ]
        if len(action.shape) == 2:
            action = action.squeeze(0)
        
        primitive = action[0]
        if self.residual:
            if self.one_hot_action:
                one_hot_idx = action[1:-2]
                obj_index = np.argmax(one_hot_idx)
                obj_pos = observation["image"][2*obj_index:2*obj_index+2]
            else:
                obj_index = action[1]
                obj_pos = observation["image"][2*obj_index:2*obj_index+2]
            if self.scale_action:
                residual = action[-2:]*self.residual_region
            else:
                residual = action[-2:]
            if self.scale_obs:
                obj_pos = obj_pos*self.image_size[0]
            pix = residual+obj_pos
            pix = np.clip(pix,[0,0],[self.image_size[0]-1,self.image_size[1]-1])
            action = np.array([primitive]+pix.tolist())
        else:
            pix = action[1:]
            if self.scale_action:
                pix = pix*self.image_size[0]
                pix = np.clip(pix,[0,0],[self.image_size[0]-1,self.image_size[1]-1])
            action = np.array([primitive]+pix.tolist())
        return action


    def step(self, action=None):
        start_time = time.time()
        print("lang goal",self.lang_goal)

        """Do pick and place motion primitive."""
        height = self.depth
        raw_action = action.copy()
        action = self.translate_action(raw_action,self.observation)
        STEP_SIM = False
        if self.last_pick_success:
            STEP_SIM = True
        else:
            if action[0]>0.5:
                STEP_SIM = False
            else:
                pix = action[1:].copy()
                STEP_SIM = self.task.p_neglect(pix,self.obj_name_to_id,self.pos_list)
            
        if not STEP_SIM:
            print("@@@@@@@@@@@@action out of meaningless")

        STEP_SIM = STEP_SIM or (not self.neglect_steps)
        if STEP_SIM :
            pick_success = self.step_simu_action(action,height)
        else:
            pick_success = False
        # position of objects
               
        obs_start_time = time.time()
        reward,done = self.get_reward(self.observation,action)
        if STEP_SIM:
            observation = self.get_observation(pick_success)
            self.observation = observation
            self.last_pick_success = pick_success
        else:
            observation = self.observation
            self.last_pick_success = pick_success
        # position of reward function
        
        self.step_count += 1
        print("action ",raw_action,"real action",action)
        print("reward",reward,"done: ",done)
        print("###############time:",time.time()-start_time,"obs time:",time.time()-obs_start_time)
        print("observation:",observation["image"], observation["object_in_hand"])
        info = self.get_info()
        return observation, reward, done,False, info
    
    def step_simu_action(self,action,height):
        ee_xyz = np.float32(pybullet.getLinkState(self.robot_id, self.tip_link_id)[0])
        max_time = 3
        pix = action[1:].copy()
        if action[0] <= 0.5:
            #pick
            self.gripper.release()
            pick_pix = pix

            pick_pix = pick_pix.astype(np.int32)
            pick_pix = np.clip(pick_pix,[0,0],[self.image_size[0]-1,self.image_size[1]-1])
            # print(pick_pix,height, BOUNDS,PIXEL_SIZE)
            pick_xyz = pix_to_xyz(pick_pix,height, BOUNDS,PIXEL_SIZE,pick=True,ee = self.ee_type)
            hover_xyz = pick_xyz.copy() + np.float32([0, 0, 0.2])
            print("steping pick: ", pick_xyz)

            moving_time = 0
            while np.linalg.norm(hover_xyz - ee_xyz) > 0.01:
                moving_time+= self.dt
                self.movep(hover_xyz,1)
                self.step_sim_and_render()
                ee_xyz = np.float32(pybullet.getLinkState(self.robot_id, self.tip_link_id)[0])
                if moving_time > max_time:
                    print("moving time out")
                    break

            moving_time = 0
            while np.linalg.norm(pick_xyz - ee_xyz) > 0.01:
                moving_time+= self.dt
                self.movep(pick_xyz)
                self.step_sim_and_render()
                ee_xyz = np.float32(pybullet.getLinkState(self.robot_id, self.tip_link_id)[0])
                if moving_time > max_time:
                    print("moving time out")
                    break

            # Pick up object.
            if self.ee_type == "suction":
                self.gripper.activate(action,self.depth)
            else:
                self.gripper.activate()
            for _ in range(240):
                self.step_sim_and_render()
            moving_time = 0
            while np.linalg.norm(hover_xyz - ee_xyz) > 0.01:
                moving_time+= self.dt
                self.movep(hover_xyz,1)
                self.step_sim_and_render()
                ee_xyz = np.float32(pybullet.getLinkState(self.robot_id, self.tip_link_id)[0])
                if moving_time > max_time:
                    print("moving time out")
                    break
            hover_xyz = np.float32([0, -0.2, 0.4])
            moving_time = 0
            while np.linalg.norm(hover_xyz - ee_xyz) > 0.01:
                self.movep(hover_xyz,1)
                self.step_sim_and_render()
                ee_xyz = np.float32(pybullet.getLinkState(self.robot_id, self.tip_link_id)[0])
                if moving_time > max_time:
                    print("moving time out")
                    break
            pick_success = int(self.gripper.check_grasp())
            
            print("pick_success",pick_success)
            return pick_success


        elif action[0] > 0.5:
            #place
            place_pix = pix
            place_pix = place_pix.astype(np.int32)
            place_pix = np.clip(place_pix,[0,0],[self.image_size[0]-1,self.image_size[1]-1])
            place_xyz = pix_to_xyz(place_pix,height, BOUNDS,PIXEL_SIZE,pick=False,ee = self.ee_type)
            print("steping: place",  place_xyz)

            # Move to place location.
            moving_time = 0
            while np.linalg.norm(place_xyz - ee_xyz) > 0.01:
                moving_time+= self.dt
                self.movep(place_xyz)
                self.step_sim_and_render()
                ee_xyz = np.float32(pybullet.getLinkState(self.robot_id, self.tip_link_id)[0])
                if moving_time>max_time:
                    print("moving time out")
                    break

            # Place down object.
            moving_time = 0
            print("place_xyz[2]",place_xyz[2],self.gripper.detect_contact())
            while (self.gripper.detect_contact()) and (place_xyz[2] > 0.03):
                
                moving_time+= self.dt
                place_xyz[2] -= 0.001
                self.movep(place_xyz)
                for _ in range(3):
                    self.step_sim_and_render()
                if moving_time>max_time:
                    print("moving time out")
                    break
            
            for _ in range(240):
                self.step_sim_and_render()
            self.gripper.release()
            place_xyz[2] = 0.2
            ee_xyz = np.float32(pybullet.getLinkState(self.robot_id, self.tip_link_id)[0])
            moving_time = 0
            while np.linalg.norm(place_xyz - ee_xyz) > 0.01:
                self.movep(place_xyz,1)
                self.step_sim_and_render()
                ee_xyz = np.float32(pybullet.getLinkState(self.robot_id, self.tip_link_id)[0])
                if moving_time > max_time:
                    print("moving time out")
                    break

            place_xyz = np.float32([0, -0.2, 0.4])
            moving_time = 0
            while np.linalg.norm(place_xyz - ee_xyz) > 0.01:
                self.movep(place_xyz,1)
                self.step_sim_and_render()
                ee_xyz = np.float32(pybullet.getLinkState(self.robot_id, self.tip_link_id)[0])
                if moving_time > max_time:
                    print("moving time out")
                    break

            return False

    def get_info(self):
        return self.info
    
    def get_observation(self,pick_success=0):

        # todo
        obs = {}
        obs["object_in_hand"] = np.array([pick_success]).astype(bool)
        lang_goal = self.task.goals
        obs['lang_goal'] = lang_goal
        self.obj_pos = self.get_object_position()

        if len(self.task.category_names) < len(self.task.config['pick'])+len(self.task.config['place']):
                keys = self.task.config['pick']
        else:
            keys = self.task.config['pick']+self.task.config['place']
        # print("hasattr",hasattr(self,'pos_list'))
        if self.pos_list is not None:
            
            last_pos = self.pos_list
            # print("last_pos",type(last_pos))
            pos_list = last_pos[:]
            changed_idxs = []
            for idx,key in enumerate(keys):
                # update position observation only if the object is moved
                _pos = xyz_to_pix(self.obj_pos[key],BOUNDS,PIXEL_SIZE)
                previous_pos = last_pos[2*idx:2*idx+2]
                if np.linalg.norm(np.array(_pos) - np.array(previous_pos)) > 5:
                    # uodate position
                    # print("update position","from",previous_pos,"to",_pos)
                    noise = np.random.normal(0, self.observation_noise/2, 2)
                    # print("noise",noise)
                    _pos = [int(_pos[0]+noise[0]),int(_pos[1]+noise[1])] 
                    pos_list[2*idx:2*idx+2] = _pos
                    changed_idxs.append(idx)
        else:
            pos_list = []
            for idx,key in enumerate(keys):
                # update position observation only if the object is moved
                _pos = xyz_to_pix(self.obj_pos[key],BOUNDS,PIXEL_SIZE)
                noise = np.random.normal(0, self.observation_noise/2, 2)
                if np.linalg.norm(noise) > 5:
                    noise = noise/np.linalg.norm(noise)*4.8

                # print("noise",noise)
                _pos = [int(_pos[0]+noise[0]),int(_pos[1]+noise[1])]
                pos_list += _pos
            
 
        self.pos_list = pos_list
        self.depth = self.get_depth(BOUNDS,PIXEL_SIZE)
        if self.image_obs:
            image = self.get_image()/255
            rgbd = np.concatenate([image,self.height_coef *self.depth[:,:,None]],axis=2)
            if not self.residual:
                rgbd_resize = cv2.resize(rgbd,dsize=None,fx=self.reshape_para,fy=self.reshape_para)
                obs["rgbd"] = rgbd_resize
            else:
                rgbd = cv2.resize(rgbd,dsize=None,fx=0.5,fy=0.5)
                if self.observation == None:
                    obs["rgbd"] = [] 
                    for i in range(len(self.task.category_names)):
                        pos = pos_list[2*i:2*i+2]
                        half_size = self.residual_region
                        patch = extract_sub_image_with_padding(rgbd, pos[0]//2, pos[1]//2, half_size*2, half_size*2)
                        obs["rgbd"].append(patch)
                    obs["rgbd"] = np.array(obs["rgbd"])
                else:
                    obs["rgbd"] = self.observation["rgbd"]
                    for idx in changed_idxs:
                        pos = pos_list[2*idx:2*idx+2]
                        half_size = self.residual_region
                        patch = extract_sub_image_with_padding(rgbd, pos[0]//2, pos[1]//2, half_size*2, half_size*2)
                        obs["rgbd"][idx] = patch




        
        if not self.scale_obs:
            obs["image"] = np.array(pos_list)
        else:
            obs["image"] = np.array(pos_list)/self.image_size[0]        
        return obs
    
    def get_reward(self,observation,action):
        reward,done = self.task.reward(observation,action,self,BOUNDS,PIXEL_SIZE,xyz_to_pix)
        return reward,done


    def reset_reward(self, obs):
        return 0,False
    def get_expert_demonstration(self):
        desired_action = self.task.exploration_action(self)
        if not self.residual:
            pri = desired_action[0]
            idx = desired_action[1:-2]
            res = desired_action[-2:]
            if self.one_hot_action:
                idx = np.argmax(idx)
            else:
                idx = idx[0]
            pos = self.observation["image"][2*idx:2*idx+2]
            pos = pos * self.image_size[0] if self.scale_obs else pos
            res = res * self.image_size[0] if self.scale_action else res
            pix = (pos + res)/self.image_size if self.scale_action else pos + res
            desired_action = np.array([pri] + pix.tolist())
            # print("desired_action",desired_action)

        return desired_action
    
def out_of_bound(pos,BOUNDS):
    _pos = pos
    # print(BOUNDS,pos)
    if (np.array(_pos)[:2] < BOUNDS[:2,0]).any() or (np.array(_pos)[:2]  > BOUNDS[:2,1]).any():
        if (np.array(_pos)[2:] < BOUNDS[2:,1]).any():
            print("out of bound")
            return True
    else:
        return False
def extract_sub_image_with_padding(image, x, y, h, w):

    
    # Calculate top left and bottom right coordinates of the sub-image
    top_left_y = y - h//2
    top_left_x = x - w//2
    bottom_right_y = y + h//2
    bottom_right_x = x + w//2
    
    # Create an output image filled with the padding value
    rgb = np.ones((h, w, 3), dtype=np.float32) * 45/255
    depth = np.ones((h, w), dtype=np.float32) * 0
    output_image = np.concatenate( (rgb, depth[:,:,None]), axis=2)

    
    # Calculate the overlapping region between the sub-image and the input image
    overlap_top_y = max(top_left_y, 0)
    overlap_top_x = max(top_left_x, 0)
    overlap_bottom_y = min(bottom_right_y, image.shape[0])
    overlap_bottom_x = min(bottom_right_x, image.shape[1])
    
    # Calculate where to place the overlapping region in the output image
    start_y = overlap_top_y - top_left_y
    start_x = overlap_top_x - top_left_x

    # print("overlap_top_x",overlap_bottom_y - overlap_top_y)
    # print("overlap_top_y",overlap_bottom_x - overlap_top_x)
    output_image[start_x:start_x + (overlap_bottom_x - overlap_top_x),
                 start_y:start_y + (overlap_bottom_y - overlap_top_y),] = image[overlap_top_x:overlap_bottom_x,overlap_top_y:overlap_bottom_y]
    
    return output_image

