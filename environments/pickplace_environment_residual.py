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
        if self.residual:
                obj_num += 1
        if image_obs:
            # self.observation_space = spaces.Dict({
            #     "image": spaces.Box(0, 1, image_size + (4,), dtype=np.float32),
            #     "lang_goal": spaces.Box(-5, 5, (512,), dtype=np.float32),
            #     "clip_image": spaces.Box(-5, 5, (512,), dtype=np.float32),
            #     "object_in_hand": spaces.Box(0, 1, (1,), dtype=np.bool8),
            #     "lang_recommendation": spaces.Box(low=np.array([0, 0, 0]),high=np.array([1, 1, 1]),dtype=np.float32),
            # })
            image_size = (int(image_size[0]*self.reshape_para),int(image_size[1]*self.reshape_para))
            
            self.observation_space = spaces.Dict({
                    "rgbd": spaces.Box(0, 1, image_size + (4,), dtype=np.float32),
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

        if not multi_discrete:
            # shit not positive
            # solution? add a object in the center? change action space to -1,1
            if self.residual:
                if self.one_hot_action:
                    low = - np.ones((1+obj_num+2,))
                    low[0:-2] = 0*low[0:-2]
                    high = np.ones((1+obj_num+2,))
                    if not scale_action:
                        high[-2] = self.image_size[0]
                        high[-1] = self.image_size[1]
                    self.action_space = spaces.Box(low=low,high=high,dtype=np.float32)
                else:
                    low = - np.ones((4,))# motion primitive/object index/position
                    high = np.ones((4,))
                    if not scale_action:
                        high[-2] = self.image_size[0]
                        high[-1] = self.image_size[1]
                    self.action_space = spaces.Box(low=low,high=high,dtype=np.float32)
            else:
                low = np.zeros((3,))
                high = np.ones((3,))
                if not scale_action:
                    high[-2] = self.image_size[0]
                    high[-1] = self.image_size[1]
                self.action_space = spaces.Box(low=low,high=high,dtype=np.float32)
                # self.action_space = spaces.Box(low=np.array([0, 0, 0]),high=np.array([1, self.image_size[0], self.image_size[1]]),dtype=np.float32)

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
                residual = action[-2:]*self.image_size[0]/2
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
        print("reward",reward)
        # print("###############time:",time.time()-start_time,"obs time:",time.time()-obs_start_time)
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
            
            if self.residual:
                pos_list += [0.5*self.image_size[0],0.5*self.image_size[1]] 
            
 
        self.pos_list = pos_list
        self.depth = self.get_depth(BOUNDS,PIXEL_SIZE)
        if self.image_obs:
            image = self.get_image()/255
            rgbd = np.concatenate([image,self.height_coef *self.depth[:,:,None]],axis=2)
            rgbd_resize = cv2.resize(rgbd,dsize=None,fx=self.reshape_para,fy=self.reshape_para)
            obs["rgbd"] = rgbd_resize
        
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
class ResSimplifyPickOrPlaceEnvWithoutLangReward(ResPickOrPlaceEnvWithoutLangReward):
    def __init__(self,
                 task = PutBlockInBowl,
                 render=False,
                 multi_discrete=False,
                 neglect_steps=True,
                 scale_obs=True,
                 scale_action=False,
                 ee = "gripper",
                 one_hot_action=False):
        super().__init__(
                        task = task,
                        image_obs=False,
                        languange_expert=False,
                        multi_discrete=multi_discrete,
                        scale_obs=scale_obs,
                        scale_action= scale_action,
                        ee = ee,
                        neglect_steps=neglect_steps,
                        render=render,
                        one_hot_action=one_hot_action
                        )
    def get_reward(self,observation,action):
        reward,done = self.task.reward(observation,action,self,BOUNDS,PIXEL_SIZE,xyz_to_pix)
        return reward,done
    def get_expert_demonstration(self):
        desired_action = self.task.exploration_action(self)
        return desired_action
    # def get_expert_demonstration(self):
    #     print("get_expert_demonstration")
    #     # print(self.observation)
    #     object_in_hand = self.last_pick_success
    #     obj_pos = self.pos_list
    #     goals = self.task.goals
    #     obj_num = len(self.task.category_names)
    #     print("obj_pos",obj_pos)
    #     print("goals",goals)
    #     if object_in_hand:
    #         picked_obj = 0
    #         for obj in self.task.config['pick']:
    #             if self.obj_pos[obj][2] >= 0.2:
    #                 picked_obj = obj
    #                 print("picked_obj",picked_obj)
            
    #         if picked_obj == self.task.config['pick'][goals[0]]:
    #             place_obj = goals[1]+len(self.task.config['pick'])
    #             res = np.array([0,0])
    #         else:
    #             place_obj = np.random.randint(0,obj_num)
    #             res = np.random.randint(0,self.image_size[0]/4,(2,))

    #         if self.one_hot_action:
    #             one_hot_idx = [1 if i== place_obj else 0  for i in range(obj_num)]
    #             idx = one_hot_idx
    #         else:
    #             idx = [place_obj]

    #         if self.scale_action:
    #             res = np.array(res)/self.image_size[0]
    #         desired_action = [1]+ idx + res.tolist()
            

    #     else:
    #         picked_obj = goals[0]
    #         if self.one_hot_action:
    #             one_hot_idx = [1 if i== picked_obj else 0  for i in range(obj_num)]
    #             idx = one_hot_idx
    #         else:
    #             idx = [picked_obj]
    #         res = np.array([0,0])
    #         if self.scale_action:
    #             res = np.array(res)/self.image_size[0]
    #         desired_action = [0]+ idx + res.tolist()

    #     print("desired_action",desired_action)
    #     return np.array(desired_action)
    
    
# class ResPickOrPlaceEnvWithoutLangReward(ResPickOrPlaceEnvWithoutLangReward):
#     def __init__(self,render=False,multi_discrete=False,neglect_steps=True,scale_obs=True,scale_action=True,one_hot_action=False):
#         task = PutBlockInBowl
#         super().__init__(task,
#                         image_obs=True,
#                         languange_expert=False,
#                         multi_discrete=multi_discrete,
#                         scale_obs=scale_obs,
#                         scale_action= scale_action,
#                         neglect_steps=neglect_steps,
#                         render=render,
#                         one_hot_action=one_hot_action
#                         )
#     def get_reward(self,observation,action):
#         phase_penalty = 0.25
#         done = False
#         reward = 0
#         last_pos = self.obj_pos
#         desired_primitive = self.last_pick_success
#         if self.last_pick_success:
#             desired_position = last_pos[self.task.config['place'][self.task.goals[1]]]
#             desired_position = xyz_to_pix(desired_position,BOUNDS,PIXEL_SIZE)
#             desired_action = [desired_primitive]+desired_position[:2]
#         else:
#             desired_position = last_pos[self.task.config['pick'][self.task.goals[0]]]
#             desired_position = xyz_to_pix(desired_position,BOUNDS,PIXEL_SIZE)
#             desired_action = [desired_primitive]+desired_position[:2]
#         y = self.last_pick_success
#         print("desired_action :",desired_action)
#         cross_ent = y*np.log(action[0]+0.001)+(1-y)*np.log(1-action[0]+0.001)
#         phase_loss = -cross_ent*phase_penalty
#         phase_loss = np.clip(phase_loss,0, phase_penalty)
#         reward -= phase_loss
#         print("phase_loss",phase_loss)
#         dis_loss = 0
#         if (action[0] < 0.5) and ( not self.last_pick_success):
#             dis = np.linalg.norm(action[1:]-desired_action[1:])
#             # dis_loss = 0.25*3/(0.5*dis+3)
#             thre = 100
#             dis_loss = - (dis/thre,1)[dis>thre]**2*0.25
#             reward += dis_loss
#         if action[0] > 0.5 and self.last_pick_success:
#             dis = np.linalg.norm(action[1:]-desired_action[1:])
#             thre = 100
#             dis_loss = - (dis/thre,1)[dis>thre]**2*0.25
#             reward += dis_loss  
#         print("dis_reward",dis_loss)        
#         done = (self.step_count >= self.max_steps)
#         _reward,_done = self.task.get_reward(self)
#         if _done:
#             self.success = True
#         else:
#             self.success = False
#         reward += _reward
#         done |= _done
#         return reward,done
#     def get_expert_demonstration(self):
#         print("get_expert_demonstration")
#         # print(self.observation)
#         object_in_hand = self.last_pick_success
#         obj_pos = self.pos_list
#         goals = self.task.goals
#         obj_num = len(self.task.category_names)
#         print("obj_pos",obj_pos)
#         print("goals",goals)
#         if object_in_hand:
#             picked_obj = 0
#             for obj in self.task.config['pick']:
#                 if self.obj_pos[obj][2] >= 0.2:
#                     picked_obj = obj
#                     print("picked_obj",picked_obj)
            
#             if picked_obj == self.task.config['pick'][goals[0]]:
#                 place_obj = goals[1]+len(self.task.config['pick'])
#                 res = np.array([0,0])
#             else:
#                 place_obj = np.random.randint(0,obj_num)
#                 res = np.random.randint(0,self.image_size[0]/4,(2,))

#             if self.one_hot_action:
#                 one_hot_idx = [1 if i== place_obj else 0  for i in range(obj_num)]
#                 idx = one_hot_idx
#             else:
#                 idx = [place_obj]

#             if self.scale_action:
#                 res = np.array(res)/self.image_size[0]
#             desired_action = [1]+ idx + res.tolist()
            

#         else:
#             picked_obj = goals[0]
#             if self.one_hot_action:
#                 one_hot_idx = [1 if i== picked_obj else 0  for i in range(obj_num)]
#                 idx = one_hot_idx
#             else:
#                 idx = [picked_obj]
#             res = np.array([0,0])
#             if self.scale_action:
#                 res = np.array(res)/self.image_size[0]
#             desired_action = [0]+ idx + res.tolist()

#         print("desired_action",desired_action)
#         return np.array(desired_action)
    
class SimplifyPickEnvWithoutLangReward(ResPickOrPlaceEnvWithoutLangReward):
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
            thre = 100
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
        # print(self.observation)
        object_in_hand = self.last_pick_success
        obj_pos = self.pos_list
        goals = self.task.goals
        print("obj_pos",obj_pos)
        print("goals",goals)
        if object_in_hand:
            # desired_position = obj_pos[6+2*goals[1]:8+2*goals[1]]
            desired_action = [1] + np.random.randint(0,self.image_size[0],(2,)).tolist()
        else:
            desired_position = obj_pos[2*goals[0]:2*goals[0]+2]
            desired_action = [0] + desired_position[:2]
        desired_action = np.array(desired_action)
        print("desired_action",desired_action)
        if self.scale_action:
            desired_action = self.scale_action(desired_action)
        return desired_action
    
    def scale_action(self,action):
        action[1:] = action[1:]/self.image_size[0]
        return action
    

    def get_info(self):

        info = {   
            'success':self.success
        }
        if 'TimeLimit.truncated' in self.info.keys():
            info['TimeLimit.truncated'] = self.info['TimeLimit.truncated']
        # self.info = {}
        return info
    def get_expert_demonstration(self):
        desired_action = self.task.exploration_action(self)
        return desired_action