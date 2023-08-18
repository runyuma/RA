from environments.environment import PickPlaceEnv,BOUNDS,PIXEL_SIZE
import numpy as np
import pybullet
import cv2
from reward.prompt import get_prompt,ask_LLM,interme_reward
from reward.prompt_withoutVild import get_description,motion_descriptor,translator
from reward.detector import VILD
from environments.utils import *
from gym import spaces
import clip
class TestEnv(PickPlaceEnv):
    def __init__(self, task):
        super().__init__(task)
        self.vlm = VILD()
    def get_reward(self,obs,action):
        time_penalty = -0.05
        pos = {}
        for key in self.obj_name_to_id.keys():
            pos[key] = pybullet.getBasePositionAndOrientation(self.obj_name_to_id[key])[0]
        moved = False
        for key in self.obj_name_to_id.keys():
            if np.linalg.norm(np.array(pos[key]) - np.array(self.obj_pos[key])) > 0.01:
                moved = True
        if not moved:
            return time_penalty,False
        else:
            img = cv2.cvtColor(obs["image"],cv2.COLOR_RGB2BGR)
            cv2.imwrite('reward/tmp/color.png',img)
            found_objects_ap,_ = self.vlm.vild_detect('reward/tmp/color.png',self.task.category_names,verbose=True)
            text = get_prompt(obs["lang_goal"],self.found_objects,found_objects_ap)
            print("text:",text)
            ans = ask_LLM(text)
            print("ans:",ans)
            if ans:
                reward = 1
                return 1,True
            else:
                reward = -time_penalty
                print("found objects:",self.found_objects)
                dense_reward = interme_reward(motion_descriptor(obs["lang_goal"],self.found_objects),action)
                reward += dense_reward
                return reward,False

            # TODO: check did the robot follow text instructions?
    def reset_reward(self,obs):
        img = cv2.cvtColor(obs["image"],cv2.COLOR_RGB2BGR)
        cv2.imwrite('reward/tmp/color.png',img)
        self.found_objects,_ = self.vlm.vild_detect('reward/tmp/color.png',self.task.category_names,verbose=True)
        print("fond objects in reset",self.found_objects)

class PickOrPlaceEnvWithoutLangReward(PickPlaceEnv):
    # choose pick/place instead of pick_and_place

    def __init__(self, task):
        super().__init__(task)
        image_size = (224,224)
        self.image_size = image_size
        #TBD: change the observation space
        self.observation_space = spaces.Dict({
            "image": spaces.Box(0, 1, image_size + (4,), dtype=np.float32),
            "lang_goal": spaces.Box(0, 49407, (77,), dtype=np.int64),
            "object_in_hand": spaces.Box(0, 1, (1,), dtype=np.bool8),
            "lang_recommendation": spaces.Box(low=np.array([0, 0, 0]),high=np.array([1, image_size[0]-1, image_size[1]-1]),dtype=np.float32),
        })
        # self.observation_space = spaces.Dict({"image":spaces.Box(0, 1, image_size + (4,), dtype=np.float32)})
        self.action_space = spaces.Box(low=np.array([0, 0, 0]),high=np.array([1, 1, 1]),dtype=np.float32)
        self.max_steps = 10
        # self.epsilon = 0.5
    def reset(self):
        self.last_pick_success = 0
        self.info = {
            "step_result":['initialization'],
            "LLM_recommendation":[],
        }
        return super().reset()



    def step(self, action=None):
        print("lang goal",self.lang_goal)
        """Do pick and place motion primitive."""
        height = self.last_observation["depth"]
        pick_success = 0
        # Move to object.
        ee_xyz = np.float32(pybullet.getLinkState(self.robot_id, self.tip_link_id)[0])
        max_time = 3
        pix = np.array(self.image_size)*action[1:].copy()
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
                self.movep(hover_xyz)
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
                self.movep(hover_xyz)
                self.step_sim_and_render()
                ee_xyz = np.float32(pybullet.getLinkState(self.robot_id, self.tip_link_id)[0])
                if moving_time > max_time:
                    print("moving time out")
                    break
            hover_xyz = np.float32([0, -0.2, 0.4])
            while np.linalg.norm(hover_xyz - ee_xyz) > 0.01:
                self.movep(hover_xyz)
                self.step_sim_and_render()
                ee_xyz = np.float32(pybullet.getLinkState(self.robot_id, self.tip_link_id)[0])
            pick_success = int(self.gripper.check_grasp())
            
            print("pick_success",pick_success)


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
                self.movep(place_xyz)
                self.step_sim_and_render()
                ee_xyz = np.float32(pybullet.getLinkState(self.robot_id, self.tip_link_id)[0])
                if moving_time>max_time:
                    print("moving time out")
                    break

            # Place down object.
            moving_time = 0
            while (not self.gripper.detect_contact()) and (place_xyz[2] > 0.03):
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
            while np.linalg.norm(place_xyz - ee_xyz) > 0.01:
                self.movep(place_xyz)
                self.step_sim_and_render()
                ee_xyz = np.float32(pybullet.getLinkState(self.robot_id, self.tip_link_id)[0])

            place_xyz = np.float32([0, -0.2, 0.4])
            while np.linalg.norm(place_xyz - ee_xyz) > 0.01:
                self.movep(place_xyz)
                self.step_sim_and_render()
                ee_xyz = np.float32(pybullet.getLinkState(self.robot_id, self.tip_link_id)[0])
        # position of objects
        pos = {}
        for key in self.obj_name_to_id.keys():
            pos[key] = pybullet.getBasePositionAndOrientation(self.obj_name_to_id[key])[0]
        

        res = get_description(self,action,pick_success,pos,last_pos=self.obj_pos)
        
        print("res:",res)
        self.info["step_result"].append(res)
        observation = self.get_observation(pick_success)
        
        reward,done = self.get_reward(observation,action,pos=pos,last_pos=self.obj_pos)
        # record as last position
        self.obj_pos = pos
        self.last_pick_success = pick_success
        self.step_count += 1
        
        return observation, reward, done, self.info
    
    def get_observation(self,pick_success=0):
        pos = {}
        for key in self.obj_name_to_id.keys():
            pos[key] = pybullet.getBasePositionAndOrientation(self.obj_name_to_id[key])[0]
        obs =  super().get_observation()
        #"image" "depth" "xyzmap" "lang_goal"
        self.last_observation = obs
        observation = {} # image(RGBD) lang_goal(encoded) object_in_hand(bool)
        img = obs["image"]/255
        img = np.concatenate([img,obs["depth"][..., None]],axis=-1)
        # print("img shape",img.shape)
        # print("depth shape",obs["depth"][..., None].shape)
        observation["image"] = img
        observation['object_in_hand'] = np.array([pick_success])
        observation['lang_goal'] = clip.tokenize(obs['lang_goal'])
        print("lang_goal",observation['lang_goal'].dtype)
        
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
        # llm_expert = True
        
        # prompting LLM
        if llm_expert:
            try:
                motion_description=motion_descriptor(self.lang_goal,pos,self.info)
                LLM_action = translator(motion_description,pos)
            except:
                LLM_action = np.zeros((3,))
        else:
            pick_idx = self.task.goals[0]
            pick_pos = pybullet.getBasePositionAndOrientation(self.obj_name_to_id[self.task.config["pick"][pick_idx]])[0]
            pix = xyz_to_pix(pick_pos,BOUNDS,PIXEL_SIZE)
            LLM_action = np.array([0,pix[0],pix[1]])
            # print("LLM_action",LLM_action)
        observation['lang_recommendation'] = np.array(LLM_action)
        self.info["LLM_recommendation"].append(LLM_action) 
        # print(self.info["step_result"],llm_expert)

        # print("lang_goal:",observation['lang_goal'].shape,observation['lang_goal'].dtype,observation['lang_goal'].max(),observation['lang_goal'].min())
        return observation
    def get_reward(self,observation,action,pos,last_pos):
        # todo cross entropy loss
        done = False
        if self.step_count >= self.max_steps:
            done = True
        time_penalty = -0.05

        moved = False
        for key in self.obj_name_to_id.keys():
            if np.linalg.norm(np.array(pos[key]) - np.array(last_pos[key])) > 0.01:
                moved = True
    
        # loss for not following the recommendation
        LLM_loss = 0
        LLM_action = self.info["LLM_recommendation"][-2]
        action_primitive = action[0]
        if action_primitive < 0.5:
            action_primitive = 0
        else:
            action_primitive = 1
        if LLM_action[0] != action_primitive:
            LLM_loss += -0.1
        else:
            dis = np.linalg.norm(np.array(self.image_size)*action[1:]-np.array(LLM_action[1:]))
            print("dis:",dis)
            LLM_loss = 0.05*5/(dis+5)
        if (LLM_action == np.zeros((3,))).all():
            LLM_loss = 0
        if not moved:
            return time_penalty+LLM_loss,done
        reward,done = self.task.get_reward(self)
        
        if done:
            print("reward:",reward,"done:",done)
            return reward,done
        else:
            reward = time_penalty+LLM_loss
            # for phase: if it already pick up the object, punish it if it not put down the object
            if self.last_pick_success == 1:
                if action[0] < 0.5:
                    reward += -0.1
            if action[0] > 0.5:
                if self.last_pick_success == 0:
                    reward += -0.1
            print("reward:",reward,"done",done)
            return time_penalty,done


    def reset_reward(self, obs):
        return 0,False