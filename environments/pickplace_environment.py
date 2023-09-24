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

    def __init__(self, task,render=False):
        super().__init__(task,render=render)
        image_size = (224,224)
        self.image_size = image_size
        self.height_coef = 20
        #TBD: change the observation space
        self.observation_space = spaces.Dict({
            "image": spaces.Box(0, 1, image_size + (4,), dtype=np.float32),
            "lang_goal": spaces.Box(-5, 5, (512,), dtype=np.float32),
            "clip_image": spaces.Box(-5, 5, (512,), dtype=np.float32),
            "object_in_hand": spaces.Box(0, 1, (1,), dtype=np.bool8),
            "lang_recommendation": spaces.Box(low=np.array([0, 0, 0]),high=np.array([1, 1, 1]),dtype=np.float32),
        })
        # self.observation_space = spaces.Dict({"image":spaces.Box(0, 1, image_size + (4,), dtype=np.float32)})
        # self.action_space = spaces.Box(low=np.array([0, 0, 0]),high=np.array([1, 1, 1]),dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([0, 0, 0]),high=np.array([1, image_size[0], image_size[1]]),dtype=np.int16)
        self.max_steps = 10
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device="cpu")

    def reset(self,seed = None):
        self.last_pick_success = 0
        self.info = {
            "step_result":['initialization'],
            "LLM_recommendation":[],
            "obj_position":[],
        }
        return super().reset()



    def step(self, action=None):
        start_time = time.time()
        print("lang goal",self.lang_goal)
        """Do pick and place motion primitive."""
        height = self.last_observation["depth"]
        pick_success = 0
        # Move to object.
        ee_xyz = np.float32(pybullet.getLinkState(self.robot_id, self.tip_link_id)[0])
        max_time = 3
        if len(action.shape) == 2:
            action = action.squeeze(0)
        pix = action[1:].copy()
        # print(np.array(self.action_space.high[1:]))
        # position of objects
        pos = {}
        TRUE_ACT = False
        for key in self.obj_name_to_id.keys():
            _pos = pybullet.getBasePositionAndOrientation(self.obj_name_to_id[key])[0]
            if key in self.task.config["pick"]:
                max_dis = 15
            if key in self.task.config["place"]:
                max_dis = 25
            _pos_pix = xyz_to_pix(_pos,BOUNDS,PIXEL_SIZE)
            # print(np.linalg.norm(_pos_pix-pix),_pos_pix,pix)
            if np.linalg.norm(_pos_pix-pix) < max_dis:
                
                TRUE_ACT = True
                break
        if self.observation["object_in_hand"][0] == 1:
            TRUE_ACT = True
        if not TRUE_ACT:
            print("@@@@@@@@@@@@action out of meaningless")


        # print("pos:",pos)
        if action[0] < 0.5 and TRUE_ACT:
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
                self.movep(hover_xyz,5)
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
                self.movep(hover_xyz,2)
                self.step_sim_and_render()
                ee_xyz = np.float32(pybullet.getLinkState(self.robot_id, self.tip_link_id)[0])
                if moving_time > max_time:
                    print("moving time out")
                    break
            hover_xyz = np.float32([0, -0.2, 0.4])
            moving_time = 0
            while np.linalg.norm(hover_xyz - ee_xyz) > 0.01:
                self.movep(hover_xyz,2)
                self.step_sim_and_render()
                ee_xyz = np.float32(pybullet.getLinkState(self.robot_id, self.tip_link_id)[0])
                if moving_time > max_time:
                    print("moving time out")
                    break
            pick_success = int(self.gripper.check_grasp())
            
            print("pick_success",pick_success)


        elif action[0] > 0.5 and TRUE_ACT:
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
                self.movep(place_xyz,5)
                self.step_sim_and_render()
                ee_xyz = np.float32(pybullet.getLinkState(self.robot_id, self.tip_link_id)[0])
                if moving_time > max_time:
                    print("moving time out")
                    break

            place_xyz = np.float32([0, -0.2, 0.4])
            moving_time = 0
            while np.linalg.norm(place_xyz - ee_xyz) > 0.01:
                self.movep(place_xyz,5)
                self.step_sim_and_render()
                ee_xyz = np.float32(pybullet.getLinkState(self.robot_id, self.tip_link_id)[0])
                if moving_time > max_time:
                    print("moving time out")
                    break
        # position of objects
        pos = {}
        for key in self.obj_name_to_id.keys():
            pos[key] = pybullet.getBasePositionAndOrientation(self.obj_name_to_id[key])[0]
        

        res = get_description(self,action,pick_success,pos,last_pos=self.obj_pos)
        
        print("res:",res)
        self.info["step_result"].append(res)
        obs_start_time = time.time()
        observation = self.get_observation(pick_success)
        
        reward,done = self.get_reward(observation,action,pos=pos,last_pos=self.obj_pos)
        # record as last position
        self.obj_pos = pos
        self.last_pick_success = pick_success
        self.step_count += 1
        print("action ",action,"reward",reward)
        print("###############time:",time.time()-start_time,"obs time:",time.time()-obs_start_time)
        print("observation:",observation)
        info = self.get_info()
        return observation, reward, done,False, info
    def get_info(self):
        return self.info
    
    def get_observation(self,pick_success=0):

        pos = {}
        for key in self.obj_name_to_id.keys():
            pos[key] = pybullet.getBasePositionAndOrientation(self.obj_name_to_id[key])[0]
        self.info["obj_position"].append(pos)
        obs =  super().get_observation()
        #"image" "depth" "xyzmap" "lang_goal"
        self.last_observation = obs
        observation = {} # image(RGBD) lang_goal(encoded) object_in_hand(bool)
        # print("obs",obs["image"].shape,obs["image"].dtype,obs["image"].max(),obs["image"].min())
        img = obs["image"]/255
        img = np.concatenate([img,self.height_coef*obs["depth"][..., None]],axis=-1)
        # print("img shape",img.shape)
        # print("depth shape",obs["depth"][..., None].shape)
        observation["image"] = img
        observation['object_in_hand'] = np.array([pick_success])
        lang_goal = clip.tokenize(obs['lang_goal'])
        with th.no_grad():
            embedding = self.clip_model.encode_text(lang_goal)
            clip_image = Image.fromarray(obs["image"])
            clip_image = self.clip_preprocess(clip_image).unsqueeze(0).to("cpu")
            clip_image = self.clip_model.encode_image(clip_image)
        # print("clip_image",clip_image.shape,clip_image.dtype,clip_image.max(),clip_image.min())
        observation['lang_goal'] = embedding.numpy()
        observation['clip_image'] = clip_image.numpy()
        # clip_image.show()
        # print("lang_goal",observation['lang_goal'].dtype)
        
        llm_expert = True
        # for saving money and time: when nothing done, do not prompt LLM
        if len(self.info["step_result"])>1:
            changed = False
            for result in self.info["step_result"][1:]:
                if result != "pick up nothing" and result != "place nothing":
                    changed = True
                    break
            if not changed:
                llm_expert = False
        else:
            llm_expert = False
        #forced to not prompt llm
        llm_expert = False
        
        # prompting LLM
        if llm_expert:
            try:
                motion_description=motion_descriptor(self.lang_goal,self.info)
                LLM_action = translator(motion_description,self.info)
                LLM_action[1:] = np.array(LLM_action[1:])/np.array(self.image_size)
            except:
                LLM_action = np.zeros((3,))
        else:
            pick_idx = self.task.goals[0]
            pick_pos = pybullet.getBasePositionAndOrientation(self.obj_name_to_id[self.task.config["pick"][pick_idx]])[0]
            pix = xyz_to_pix(pick_pos,BOUNDS,PIXEL_SIZE)
            LLM_action = np.array([0,pix[0]/self.image_size[0],pix[1]/self.image_size[1]])
            # print("LLM_action",LLM_action)
        observation['lang_recommendation'] = np.array(LLM_action)
        self.info["LLM_recommendation"].append(LLM_action)
        self.observation = observation 
        # print("image",img.shape,observation["image"].dtype,img.max(),img.min())
        # print(self.info["step_result"],llm_expert)

        # print("lang_goal:",observation['lang_goal'].shape,observation['lang_goal'].dtype,observation['lang_goal'].max(),observation['lang_goal'].min())
        return observation
    def super_get_observation(self):
        return super().get_observation()
    def get_reward(self,observation,action,pos,last_pos):
        # todo cross entropy loss
        phase_penalty = 0.2
        damage_penalty = 0.25
        done = False
        if self.step_count >= self.max_steps:
            done = True
        time_penalty = -0.05

        moved = False
        for key in self.obj_name_to_id.keys():
            if np.linalg.norm(np.array(pos[key]) - np.array(last_pos[key])) > 0.01:
                moved = True
            if out_of_bound(pos[key],BOUNDS):
                done = True
                return -damage_penalty,done
    
        # loss for not following the recommendation
        LLM_loss = 0
        # LLM_action = self.info["LLM_recommendation"][-2]
        # action_primitive = action[0]
        # if action_primitive < 0.5:
        #     action_primitive = 0
        # else:
        #     action_primitive = 1
        # if LLM_action[0] != action_primitive:
        #     LLM_loss += -phase_penalty
        # else:
        #     dis = np.linalg.norm(np.array(self.image_size)*action[1:]-np.array(self.image_size)*np.array(LLM_action[1:]))
        #     print("dis:",dis)
        #     LLM_loss = 0.05*5/(dis+5)
        # if (LLM_action == np.zeros((3,))).all():
        #     LLM_loss = 0
        if not moved:
            return time_penalty+LLM_loss,done
        reward,_done = self.task.get_reward(self)
        done |= _done
        
        if done:
            print("reward:",reward,"done:",done)
            return reward,done
        else:
            reward = time_penalty+LLM_loss
            # for phase: if it already pick up the object, punish it if it not put down the object
            if self.last_pick_success == 1:
                if action[0] < 0.5:
                    reward += -phase_penalty
            if action[0] > 0.5:
                if self.last_pick_success == 0:
                    reward += -phase_penalty
            print("reward:",reward,"done",done)
            return time_penalty,done


    def reset_reward(self, obs):
        return 0,False

    @limit_decor(10, 1.)
    def prompt_llm_func(self):
        motion_description=motion_descriptor(self.lang_goal,self.info)
        LLM_action = translator(motion_description,self.info)
        LLM_action[1:] = np.array(LLM_action[1:])/np.array(self.image_size)*self.action_space.high[1:]
        return LLM_action
    
    def prompt_llm(self):
        try:
            # print("########################prompting LLM")
            LLM_action = self.prompt_llm_func()
            if LLM_action is None:
                LLM_action = np.zeros((3,))
        except:
            LLM_action = np.zeros((3,))
        return LLM_action
    
def out_of_bound(pos,BOUNDS):
    _pos = pos
    # print(BOUNDS,pos)
    if (np.array(_pos)[:2] < BOUNDS[:2,0]).any() or (np.array(_pos)[:2]  > BOUNDS[:2,1]).any():
        if (np.array(_pos)[2:] < BOUNDS[2:,1]).any():
            print("out of bound")
            return True
    else:
        return False
class SimplifyPickOrPlaceEnvWithoutLangReward(PickOrPlaceEnvWithoutLangReward):
    def __init__(self,render=False):
        task = PutBlockInBowl
        super().__init__(task,render)
        image_size = (224,224)
        self.image_size = image_size
        self.height_coef = 20
        #TBD: change the observation space
        obj_num = len(self.task.category_names)
        goal_num = 2
        self.observation_space = spaces.Dict({
            "image": spaces.Box(0, image_size[0], (obj_num*2,), dtype=np.int32),
            "lang_goal": spaces.Box(0, obj_num, (goal_num,), dtype=np.int32),
            "object_in_hand": spaces.Box(0, 1, (1,), dtype=np.bool8),
        })
        # self.observation_space = spaces.Dict({"image":spaces.Box(0, 1, image_size + (4,), dtype=np.float32)})
        # self.action_space = spaces.Box(low=np.array([0, 0, 0]),high=np.array([1, 1, 1]),dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([0, 0, 0]),high=np.array([1, image_size[0], image_size[1]]),dtype=np.float32)
        self.max_steps = 10
        # self.epsilon = 0.5
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device="cpu")

    def get_observation(self,pick_success=0):

        pos = {}
        for key in self.obj_name_to_id.keys():
            pos[key] = pybullet.getBasePositionAndOrientation(self.obj_name_to_id[key])[0]
        self.info["obj_position"].append(pos)
        pos_list = []
        for key in self.task.config['pick']:
            pos_list += xyz_to_pix(pos[key],BOUNDS,PIXEL_SIZE)
        for key in self.task.config['place']:
            pos_list += xyz_to_pix(pos[key],BOUNDS,PIXEL_SIZE)
        
        # print("task config",self.task.config)
        obs =  super().super_get_observation()
        #"image" "depth" "xyzmap" "lang_goal"
        self.last_observation = obs
        observation = {} # image(RGBD) lang_goal(encoded) object_in_hand(bool)
        observation['image'] = pos_list #pesudo image with position sequence
        observation['object_in_hand'] = np.array([pick_success])
        lang_goal = self.task.goals
        observation['lang_goal'] = lang_goal
        self.observation = observation
        return observation

        # clip_image.show()
        # print("lang_goal",observation['lang_goal'].dtype)
    def get_reward(self,observation,action,pos,last_pos):
        phase_penalty = 0.25
        damage_penalty = 0.25
        done = False
        time_penalty = -0.05
        reward = 0
        desired_primitive = self.last_pick_success
        if self.last_pick_success:
            desired_position = last_pos[self.task.config['place'][self.task.goals[1]]]
            desired_position = xyz_to_pix(desired_position,BOUNDS,PIXEL_SIZE)
            desired_action = [desired_primitive]+desired_position[:2]
        else:
            desired_position = last_pos[self.task.config['pick'][self.task.goals[0]]]
            desired_position = xyz_to_pix(desired_position,BOUNDS,PIXEL_SIZE)
            desired_action = [desired_primitive]+desired_position[:2]
        y = self.last_pick_success
        print("desired_action",desired_action)
        cross_ent = y*np.log(action[0]+0.001)+(1-y)*np.log(1-action[0]+0.001)
        phase_loss = -cross_ent*phase_penalty
        phase_loss = np.clip(phase_loss,0, phase_penalty)
        reward -= phase_loss
        print("phase_loss",phase_loss)
        dis_loss = 0
        if (action[0] > 0.5)== self.last_pick_success:
            dis = np.linalg.norm(action[1:]-desired_action[1:])
            dis_loss = 0.25*3/(0.5*dis+3)
            reward += dis_loss
        print("dis_reward",dis_loss)        
        done = (self.step_count >= self.max_steps)
        _reward,_done = self.task.get_reward(self)
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
            desired_action = [1]+desired_position[:2]
        else:
            desired_position = obj_pos[2*goals[0]:2*goals[0]+2]
            desired_action = [0]+desired_position[:2]
        print("desired_action",desired_action)
        return desired_action
        
class SimplifyPickEnvWithoutLangReward(PickOrPlaceEnvWithoutLangReward):
    def __init__(self,render=False,multi_discrete=False,scale_obs=False):
        task = PickBlock
        super().__init__(task,render)
        
        obj_num = len(self.task.category_names)
        goal_num = 1
        self.scale_obs = scale_obs
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
        # self.observation_space = spaces.Dict({"image":spaces.Box(0, 1, image_size + (4,), dtype=np.float32)})
        # self.action_space = spaces.Box(low=np.array([0, 0, 0]),high=np.array([1, 1, 1]),dtype=np.float32)
        if not multi_discrete:
            self.action_space = spaces.Box(low=np.array([0, 0, 0]),high=np.array([1, self.image_size[0], self.image_size[1]]),dtype=np.float32)
        else:
            self.action_space = spaces.MultiDiscrete([1, self.image_size[0], self.image_size[1]])
        self.max_steps = 10


    def get_observation(self,pick_success=0):

        pos = {}
        for key in self.obj_name_to_id.keys():
            pos[key] = pybullet.getBasePositionAndOrientation(self.obj_name_to_id[key])[0]
        self.info["obj_position"].append(pos)
        pos_list = []
        for key in self.task.config['pick']:
            pos_list += xyz_to_pix(pos[key],BOUNDS,PIXEL_SIZE)
        # for key in self.task.config['place']:
        #     pos_list += xyz_to_pix(pos[key],BOUNDS,PIXEL_SIZE)
        
        # print("task config",self.task.config)
        obs =  super().super_get_observation()
        #"image" "depth" "xyzmap" "lang_goal"
        self.last_observation = obs
        observation = {} # image(RGBD) lang_goal(encoded) object_in_hand(bool)
        if not self.scale_obs:
            observation['image'] = pos_list
        else:
            observation['image'] = np.array(pos_list)/self.image_size[0]
        #pesudo image with position sequence
        observation['object_in_hand'] = np.array([pick_success])
        lang_goal = self.task.goals
        observation['lang_goal'] = lang_goal
        self.observation = observation
        return observation

        # clip_image.show()
        # print("lang_goal",observation['lang_goal'].dtype)
    def get_reward(self,observation,action,pos,last_pos):
        phase_penalty = 0.25
        damage_penalty = 0.25
        done = False
        time_penalty = -0.1
        reward = 0
        desired_primitive = self.last_pick_success
        if self.last_pick_success:
            desired_position = [0,0]
            desired_position = xyz_to_pix(desired_position,BOUNDS,PIXEL_SIZE)
            desired_action = [desired_primitive]+desired_position[:2]
        else:
            desired_position = last_pos[self.task.config['pick'][self.task.goals[0]]]
            desired_position = xyz_to_pix(desired_position,BOUNDS,PIXEL_SIZE)
            desired_action = [desired_primitive]+desired_position[:2]
        y = self.last_pick_success
        print("desired_action",desired_action)
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