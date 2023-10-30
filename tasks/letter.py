import numpy as np
from environments.environment import BOUNDS,PIXEL_SIZE,COLORS
import pybullet
from tasks.task import Task
import string
import random
import tempfile
import os
class PutLetterOontheBowl(Task):
    """Put Blocks in Bowl base class and task."""

    def __init__(self,colors,bounds):
        super().__init__(
            config = {'pick':  ['yellow Letter O', 'green Letter O', 'blue Letter O'],'place': ['yellow bowl', 'green bowl', 'blue bowl']},
            lang_template = "put the {pick} on a {place}",
            goal_num = 2,
            colors = colors,
            bounds = bounds,
        )
        self.residual_region = 28//2
    def set_env(self,env,goals,obj_names,pos):
        pick_idx,place_idx = goals
        obj_name_to_id = {}
        env.lang_goal = self.lang_template.format(pick=self.config["pick"][pick_idx], place=self.config["place"][place_idx])
        for obj_name in obj_names:
            object_type = obj_name.split(" ")[1]
            rand_xyz = pos.pop(0)
            object_position = rand_xyz.squeeze()
            if object_type != "bowl":
                object_color = self.colors[obj_name.split(" ")[0]]
                shape = "environments/kitting/16.obj"
                object_position[2] = 0
                template = 'environments/kitting/object-template.urdf'
                full_template_path = "environments/kitting/object-template.urdf"
                # 00R O1A 05T 12G 13V 14E 15L 16O 19M
                scale = [0.0045, 0.0045, 0.001]
                replace = {'FNAME': (shape,), 'SCALE': scale, 'COLOR': (0.2, 0.2, 0.2)}
                with open(full_template_path, 'r') as file:
                    fdata = file.read()
                for field in replace:
                    for i in range(len(replace[field])):
                        fdata = fdata.replace(f'{field}{i}', str(replace[field][i]))
                # print(fdata)
                alphabet = string.ascii_lowercase + string.digits
                rname = ''.join(random.choices(alphabet, k=16))
                tmpdir = tempfile.gettempdir()
                template_filename = os.path.split(template)[-1]
                fname = os.path.join(tmpdir, f'{template_filename}.{rname}')
                with open(fname, 'w') as file:
                    file.write(fdata)

                object_id = pybullet.loadURDF(fname, object_position, useFixedBase=0)
            elif object_type == "bowl":
                object_color = self.colors[obj_name.split(" ")[0]]
                object_position[2] = 0.003
                object_id = pybullet.loadURDF("environments/bowl/bowl.urdf", object_position, useFixedBase=1)
            pybullet.changeVisualShape(object_id, -1, rgbaColor=object_color)
            obj_name_to_id[obj_name] = object_id
        env.obj_name_to_id = obj_name_to_id
    def sample_goals(self):
        pick_idx =np.random.choice(len(self.config["pick"]))
        place_idx = np.random.choice(len(self.config["place"]))
        goals = [pick_idx,place_idx]
        return goals
        
    def get_reward(self,env):
        pick_idx,place_idx = self.goals
        pick_pos = pybullet.getBasePositionAndOrientation(env.obj_name_to_id[self.config["pick"][pick_idx]])[0]
        place_pos = pybullet.getBasePositionAndOrientation(env.obj_name_to_id[self.config["place"][place_idx]])[0]
        if np.linalg.norm(np.array(pick_pos)[:2] - np.array(place_pos)[:2]) < self.pos_eps:
        
            return 1,True
        else:
            return 0,False
    def exploration_action(self,env):
        print("get_expert_demonstration")
        # print(self.observation)
        obs = env.observation
        object_in_hand = env.last_pick_success
        obj_pos = env.pos_list
        goals = self.goals
        obj_num = len(self.category_names)
        print("obj_pos",obj_pos)
        print("goals",goals)
        prim = 1 if object_in_hand else 0
        idx = goals[prim]
        
        if object_in_hand:
            picked_obj = 0
            for obj in self.config['pick']:
                if env.obj_pos[obj][2] >= 0.2:
                    picked_obj = obj
                    print("picked_obj",picked_obj)
            if picked_obj == self.config['pick'][goals[0]]:
                obj_idx = goals[1]+len(self.config['pick'])
                angle = np.random.uniform(0,2*np.pi)
                r = np.random.uniform(0,2)
                res = np.array([np.cos(angle),np.sin(angle)])*r
            else:
                obj_idx = np.random.randint(0,obj_num)
                res = np.random.randint(0,env.image_size[0]/4,(2,))
        else:
            obj_idx = goals[0]
            angle = np.random.uniform(0,2*np.pi)
            r = np.random.uniform(7,15)#todo
            res = np.array([np.cos(angle),np.sin(angle)])*r
        if env.scale_action:
            res = np.array(res)/env.residual_region
        if env.one_hot_action:
            one_hot_idx = [1 if i== obj_idx else 0  for i in range(obj_num)]
            idx = one_hot_idx
            desired_action = [prim]+ idx + res.tolist()
        else:
            desired_action = [prim,obj_idx]+res.tolist()


        
        # if object_in_hand:
        #     picked_obj = 0
        #     for obj in self.config['pick']:
        #         if env.obj_pos[obj][2] >= 0.2:
        #             picked_obj = obj
        #             print("picked_obj",picked_obj)
            
        #     if picked_obj == self.config['pick'][goals[0]]:
        #         place_obj = goals[1]+len(self.config['pick'])
        #         angle = np.random.uniform(0,2*np.pi)
        #         r = np.random.uniform(0,2)
        #         res = np.array([np.cos(angle),np.sin(angle)])*r
        #         # res = np.array([0,0])
        #     else:
        #         place_obj = np.random.randint(0,obj_num)
        #         res = np.random.randint(0,env.image_size[0]/4,(2,))

        #     if env.one_hot_action:
        #         one_hot_idx = [1 if i== place_obj else 0  for i in range(obj_num+1)]
        #         idx = one_hot_idx
        #     else:
        #         idx = [place_obj]

        #     if env.scale_action:
        #         res = np.array(res)/env.image_size[0]
        #     desired_action = [1]+ idx + res.tolist()
            

        # else:
        #     picked_obj = goals[0]
        #     if env.one_hot_action:
        #         one_hot_idx = [1 if i== picked_obj else 0  for i in range(obj_num+1)]
        #         idx = one_hot_idx
        #     else:
        #         idx = [picked_obj]
        #     angle = np.random.uniform(0,2*np.pi)
        #     r = 0
        #     res = np.array([np.cos(angle),np.sin(angle)])*r
        #     if env.scale_action:
        #         res = np.array(res)/env.image_size[0]
        #     desired_action = [0]+ idx + res.tolist()

        print("desired_action",desired_action)
        return np.array(desired_action) 
        
    def reward(self,observation,action,env,BOUNDS,PIXEL_SIZE,xyz_to_pix):
        phase_penalty = 0.25
        done = False
        reward = 0
        last_pos = env.obj_pos
        desired_primitive = env.last_pick_success
        if env.last_pick_success:
            desired_position = last_pos[self.config['place'][self.goals[1]]]
            desired_position = xyz_to_pix(desired_position,BOUNDS,PIXEL_SIZE)
            desired_action = [desired_primitive]+desired_position[:2]
        else:
            desired_position = last_pos[self.config['pick'][self.goals[0]]]
            desired_position = xyz_to_pix(desired_position,BOUNDS,PIXEL_SIZE)
            desired_action = [desired_primitive]+desired_position[:2]
        y = env.last_pick_success
        cross_ent = y*np.log(action[0]+0.001)+(1-y)*np.log(1-action[0]+0.001)
        phase_loss = -cross_ent*phase_penalty
        phase_loss = np.clip(phase_loss,0, phase_penalty)
        reward -= phase_loss
        dis_loss = 0
        if (action[0] < 0.5) and ( not env.last_pick_success):
            dis = np.linalg.norm(action[1:]-desired_action[1:])
            # dis_loss = 0.25*3/(0.5*dis+3)
            thre = 100
            dis_loss = - (dis/thre,1)[dis>thre]**2*0.25
            reward += dis_loss
        if action[0] > 0.5 and env.last_pick_success:
            dis = np.linalg.norm(action[1:]-desired_action[1:])
            thre = 100
            dis_loss = - (dis/thre,1)[dis>thre]**2*0.25
            reward += dis_loss  
        print("desired_action :",desired_action,"phase_loss",phase_loss,"dis_reward",dis_loss)        
        done = (env.step_count >= env.max_steps)
        _reward,_done = self.get_reward(env)
        if _done:
            env.success = True
        else:
            env.success = False
        reward += _reward
        done |= _done
        return reward,done