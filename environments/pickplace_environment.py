from environments.environment import PickPlaceEnv,BOUNDS,PIXEL_SIZE
from tasks.task import PutBlockInBowl,PickBlock
from environments.timer import limit_decor
import numpy as np
import pybullet
import cv2
from environments.utils import *
# from gym import spaces
from gymnasium import spaces
import torch as th
import time
from PIL import Image
import time


class PickOrPlaceEnvWithoutLangReward(PickPlaceEnv):
    # choose pick/place instead of pick_and_place

    def __init__(self, task,image_obs,languange_expert,multi_discrete,scale_obs,scale_action,neglect_steps,render=False):
        super().__init__(task,render=render)
        image_size = (224,224)
        self.image_size = image_size
        self.height_coef = 20

        self.image_obs = image_obs
        self.neglect_steps = neglect_steps
        self.scale_obs = scale_obs
        self.scale_action = scale_action

        obj_num = len(self.task.category_names)
        goal_num = self.task.goal_num
        
        if image_obs:
            self.observation_space = spaces.Dict({
                "image": spaces.Box(0, 1, image_size + (4,), dtype=np.float32),
                "lang_goal": spaces.Box(-5, 5, (512,), dtype=np.float32),
                "clip_image": spaces.Box(-5, 5, (512,), dtype=np.float32),
                "object_in_hand": spaces.Box(0, 1, (1,), dtype=np.bool8),
                "lang_recommendation": spaces.Box(low=np.array([0, 0, 0]),high=np.array([1, 1, 1]),dtype=np.float32),
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

        if not multi_discrete:
            if not scale_action:
                self.action_space = spaces.Box(low=np.array([0, 0, 0]),high=np.array([1, self.image_size[0], self.image_size[1]]),dtype=np.float32)
            else:
                self.action_space = spaces.Box(low=np.array([0, 0, 0]),high=np.array([1, 1, 1]),dtype=np.float32)
        else:
            self.action_space = spaces.MultiDiscrete([2, self.image_size[0], self.image_size[1]])
        self.max_steps = 10 # task.max_steps

        if languange_expert:
            import clip
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device="cpu")

    def reset(self,seed = None):
        self.last_pick_success = 0
        self.info = {
            "step_result":['initialization'],
            "LLM_recommendation":[],
            "obj_position":[],
        }
        # get observation && ground truth position
        obs,info = super().reset()
        self.observation = obs
        return obs,info


    def step(self, action=None):
        start_time = time.time()
        print("lang goal",self.lang_goal)

        """Do pick and place motion primitive."""
        height = self.depth
        if len(action.shape) == 2:
            action = action.squeeze(0)
        
        STEP_SIM = False
        if self.last_pick_success:
            STEP_SIM = True
        else:
            pix = action[1:].copy()
            for key in self.obj_name_to_id.keys():
                if key in self.task.config["pick"]:
                    max_dis = 15
                if key in self.task.config["place"]:
                    max_dis = 25
                _pos_pix = self.pos_list[2*self.task.category_names.index(key):2*self.task.category_names.index(key)+2]
                print("distance with ", key,": ",np.linalg.norm(_pos_pix-pix),_pos_pix,pix)
                if np.linalg.norm(_pos_pix-pix) < max_dis:
                    STEP_SIM = True
                    break
            
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
        print("action ",action,"reward",reward)
        print("###############time:",time.time()-start_time,"obs time:",time.time()-obs_start_time)
        print("observation:",observation)
        info = self.get_info()
        return observation, reward, done,False, info
    
    def step_simu_action(self,action,height):
        ee_xyz = np.float32(pybullet.getLinkState(self.robot_id, self.tip_link_id)[0])
        max_time = 3
        pix = action[1:].copy()
        if action[0] < 0.5:
            #pick
            self.gripper.release()
            pick_pix = pix

            pick_pix = pick_pix.astype(np.int32)
            pick_pix = np.clip(pick_pix,[0,0],[self.image_size[0]-1,self.image_size[1]-1])
            # print(pick_pix,height, BOUNDS,PIXEL_SIZE)
            pick_xyz = pix_to_xyz(pick_pix,height, BOUNDS,PIXEL_SIZE,pick=True)
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
            place_xyz = pix_to_xyz(place_pix,height, BOUNDS,PIXEL_SIZE,pick=False)
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
            self.gripper.release()
            for _ in range(240):
                self.step_sim_and_render()
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
        obs["object_in_hand"] = np.array([pick_success])
        lang_goal = self.task.goals
        obs['lang_goal'] = lang_goal
        self.obj_pos = self.get_object_position()
        pos_list = []
        for key in self.task.config['pick']:
            pos_list += xyz_to_pix(self.obj_pos[key],BOUNDS,PIXEL_SIZE)
        self.pos_list = pos_list
        self.depth = self.get_depth(BOUNDS,PIXEL_SIZE)
        if self.image_obs:
            image = self.get_image()
            obs["image"] = image
        else:
            if not self.scale_obs:
                obs["image"] = np.array(pos_list)
            else:
                obs["image"] = np.array(pos_list)/self.image_size[0]        
        return obs
    
    def get_reward(self,observation,action,pos,last_pos):
        raise NotImplementedError


    def reset_reward(self, obs):
        return 0,False
    
def out_of_bound(pos,BOUNDS):
    _pos = pos
    # print(BOUNDS,pos)
    if (np.array(_pos)[:2] < BOUNDS[:2,0]).any() or (np.array(_pos)[:2]  > BOUNDS[:2,1]).any():
        if (np.array(_pos)[2:] < BOUNDS[2:,1]).any():
            print("out of bound")
            return True
    else:
        return False

class SimplifyPickEnvWithoutLangReward(PickOrPlaceEnvWithoutLangReward):
    def __init__(self,render=False,multi_discrete=False,neglect_steps=True):
        task = PickBlock
        super().__init__(task,
                        image_obs=False,
                        languange_expert=False,
                        multi_discrete=multi_discrete,
                        scale_obs=True,
                        scale_action= False,
                        neglect_steps=neglect_steps,
                        render=render
                        )
        
    def get_reward(self,observation,action):
        phase_penalty = 0.25
        damage_penalty = 0.25
        done = False
        time_penalty = -0.1
        reward = 0
        last_pos = self.obj_pos
        desired_primitive = self.last_pick_success
        if self.last_pick_success:
            desired_position = [0,0]
            print("desired_position",desired_position)
            desired_position = xyz_to_pix(desired_position,BOUNDS,PIXEL_SIZE)
            desired_action = [desired_primitive]+desired_position[:2]
        else:
            desired_position = last_pos[self.task.config['pick'][self.task.goals[0]]]
            desired_position = xyz_to_pix(desired_position,BOUNDS,PIXEL_SIZE)
            desired_action = [desired_primitive]+desired_position[:2]
        y = self.last_pick_success
        print("desired_action :",desired_action)
        cross_ent = y*np.log(action[0]+0.001)+(1-y)*np.log(1-action[0]+0.001)
        phase_loss = -cross_ent*phase_penalty
        phase_loss = np.clip(phase_loss,0, phase_penalty)
        reward -= phase_loss
        print("phase_loss",phase_loss)
        dis_loss = 0
        if (action[0] < 0.5) and ( not self.last_pick_success):
            dis = np.linalg.norm(action[1:]-desired_action[1:])
            # dis_loss = 0.25*3/(0.5*dis+3)
            thre = 50
            dis_loss = - (dis/thre,1)[dis>thre]**2*0.25
            reward += dis_loss
        print("dis_reward",dis_loss)        
        done = (self.step_count >= self.max_steps)
        _reward,_done = self.task.get_reward(self)
        if _done:
            self.success = True
        else:
            self.success = False
        reward += _reward
        done |= _done
        return reward,done
    
    def get_expert_demonstration(self):
        print("get_expert_demonstration")
        print(self.observation)
        object_in_hand = self.observation['object_in_hand'][0]
        obj_pos = self.observation['image']
        goals = self.observation['lang_goal']
        if object_in_hand:
            desired_position = obj_pos[6+2*goals[1]:8+2*goals[1]]
            desired_action = [1] + np.random.randint(0,self.image_size[0],(2,))
        else:
            desired_position = obj_pos[2*goals[0]:2*goals[0]+2]
            desired_action = [0] + desired_position[:2]
        print("desired_action",desired_action)
        return desired_action
    def get_info(self):

        info = {   
            'success':self.success
        }
        if 'TimeLimit.truncated' in self.info.keys():
            info['TimeLimit.truncated'] = self.info['TimeLimit.truncated']
        # self.info = {}
        return info