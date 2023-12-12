import numpy as np
import copy
from environments.environment import BOUNDS,PIXEL_SIZE,COLORS
import pybullet
from tasks.task import Task
import string
import random
import tempfile
import os
from prompt.low_level import generate_pick_probability_map,calculate_placement_probability
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
class PutLetterontheBowl(Task):
    """Put Blocks in Bowl base class and task."""

    def __init__(self,colors,bounds, config = None):
        if config is None:
            config = {'pick':  ['Letter O', 'Letter R', 'Letter V'],'place': ['yellow bowl', 'green bowl', 'blue bowl']}
        super().__init__(
            config = config,
            lang_template = "put the {pick} on a {place}",
            goal_num = 2,
            colors = colors,
            bounds = bounds,
        )
        self.residual_region = 28//2
        self.category_names = []
        for i in range(len(self.global_config["pick"])):
            self.category_names.append(self.global_config["pick"][i])
        for i in range(len(self.global_config["place"])):    
            if self.global_config["place"][i] not in self.category_names:
                self.category_names.append(self.global_config["place"][i])
    def get_available_colors(self):
        return ["yellow","green","blue"]
    
    def reset(self,env):
        # Load objects according to config.
        self.config = self.global_config
        self.config["pick"] = np.random.permutation(self.config["pick"]).tolist()
        # print("config",self.config)
        self.category_names = []
        for i in range(len(self.config["pick"])):
            self.category_names.append(self.config["pick"][i])
        for i in range(len(self.config["place"])):    
            if self.config["place"][i] not in self.category_names:
                self.category_names.append(self.config["place"][i])
        timeout = True
        obj_names = list(self.config["pick"]) + list(self.config["place"])
        while timeout:
            # print("init scene")
            pos,goals,timeout = self.plan_scene()

        self.goals = goals
        self.set_env(env,goals,obj_names,pos)

    def set_env(self,env,goals,obj_names,pos):
        pick_idx,place_idx = goals
        obj_name_to_id = {}
        env.lang_goal = self.lang_template.format(pick=self.config["pick"][pick_idx], place=self.config["place"][place_idx])
        self.local_config = copy.copy(self.config)
        self.local_config["pick"] = []
        for obj_name in obj_names:
            object_type = obj_name.split(" ")[1]
            rand_xyz = pos.pop(0)
            object_position = rand_xyz.squeeze()
            if object_type != "bowl":
                _colors = self.get_available_colors()
                object_color = np.random.choice(_colors)
                self.local_config["pick"].append(object_color+" "+obj_name)
                object_color = self.colors[object_color]
                if object_type == "O":
                    shape = "environments/kitting/16.obj"
                elif object_type == "L":
                    shape = "environments/kitting/15.obj"
                elif object_type == "V":
                    shape = "environments/kitting/13.obj"
                elif object_type == "G":
                    shape = "environments/kitting/12.obj"
                elif object_type == "T":
                    shape = "environments/kitting/05.obj"
                elif object_type == "R":
                    shape = "environments/kitting/00.obj"
                elif object_type == "A":
                    shape = "environments/kitting/01.obj"
                object_position[2] = 0
                template = 'environments/kitting/object-template.urdf'
                full_template_path = "environments/kitting/object-template.urdf"
                # 00R O1A 05T 12G 13V 14E 15L 16O 19M
                scale = [0.005, 0.005, 0.001]
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
                theta = np.random.uniform(0, 2 * np.pi)
                object_orientation = pybullet.getQuaternionFromEuler([0, 0, theta])
                object_id = pybullet.loadURDF(fname, object_position,object_orientation, useFixedBase=0)
            elif object_type == "bowl":
                object_color = self.colors[obj_name.split(" ")[0]]
                object_position[2] = 0.003
                object_id = pybullet.loadURDF("environments/bowl/bowl.urdf", object_position, useFixedBase=1)
            pybullet.changeVisualShape(object_id, -1, rgbaColor=object_color)
            obj_name_to_id[obj_name] = object_id
        env.obj_name_to_id = obj_name_to_id
    def sample_goals(self):
        pick_idx =np.random.choice(len(self.config["pick"]),
                                #    p=[0.25,0.5,0.25]
                                   )
        place_idx = np.random.choice(len(self.config["place"]))
        goals = [pick_idx,place_idx]
        return goals
        
    def get_reward(self,env):
        pick_idx,place_idx = self.goals
        # print("pick_idx",pick_idx,"place_idx",place_idx)
        pick_pos = pybullet.getBasePositionAndOrientation(env.obj_name_to_id[self.config["pick"][pick_idx]])[0]
        place_pos = pybullet.getBasePositionAndOrientation(env.obj_name_to_id[self.config["place"][place_idx]])[0]
        if np.linalg.norm(np.array(pick_pos)[:2] - np.array(place_pos)[:2]) < self.pos_eps:
        
            return 1,True
        else:
            return 0,False
    def exploration_action(self,env,deterministic=False):
        # print("get_expert_demonstration")
        # print(self.observation)
        obs = env.observation
        object_in_hand = env.last_pick_success
        obj_pos = env.pos_list
        goals = self.goals
        obj_num = len(self.category_names)
        picked_obj = None
        for obj in self.config['pick']:
            if env.obj_pos[obj][2] >= 0.2:
                picked_obj = self.config['pick'].index(obj)
                # print("picked_obj",picked_obj)
        
        if object_in_hand and picked_obj == goals[0]:
            prim = 1
            placed_obj = goals[1]+len(self.config['pick'])
            one_hot_idx = [1 if i== placed_obj else 0  for i in range(obj_num)]
            # print(obs['rgbd'].shape)
            pick_img = (255*obs['rgbd'][picked_obj][:,:,:3]).astype(np.uint8)
            plac_img = (255*obs['rgbd'][placed_obj][:,:,:3]).astype(np.uint8)
            try:
                prob_map = calculate_placement_probability(pick_img,plac_img)
            except:
                prob_map = np.ones((plac_img.shape[0],plac_img.shape[1]))/np.sum(np.ones((plac_img.shape[0],plac_img.shape[1])))
            pix = sample_position(prob_map,MAX=deterministic)
            # import matplotlib
            # matplotlib.use('TKAgg')
            # import matplotlib.pyplot as plt
            # plt.subplot(1,3,1)
            # plt.imshow(prob_map)
            # plt.subplot(1,3,2)
            # plt.imshow(pick_img)
            # plt.subplot(1,3,3)
            # plt.imshow(plac_img)
            # plt.show()

            
        else:
            prim = 0
            pick_obj = goals[0]
            one_hot_idx = [1 if i== pick_obj else 0  for i in range(obj_num)]
            pick_img = (255*obs['rgbd'][pick_obj][:,:,:3]).astype(np.uint8)
            try:
                prob_map = generate_pick_probability_map(pick_img,deterministic=deterministic)
            except:
                prob_map = np.ones((pick_img.shape[0],pick_img.shape[1]))/np.sum(np.ones((pick_img.shape[0],pick_img.shape[1])))
            pix = sample_position(prob_map,MAX=deterministic)
            # import matplotlib
            # matplotlib.use('TKAgg')
            # import matplotlib.pyplot as plt
            # plt.subplot(1,2,1)
            # plt.imshow(prob_map)
            # plt.subplot(1,2,2)
            # plt.imshow(pick_img)
            # plt.show()
        res = (np.array(pix)-np.array([28,28])/2)
        if env.scale_action:
            res = np.array(res)/env.residual_region
        desired_action = [prim]+ one_hot_idx + res.tolist()            

        return np.array(desired_action) 
    def p_neglect(self,pix,obj_name_to_id,pos_list):
        for key in obj_name_to_id.keys():
            if key in self.config["pick"] : 
                max_dis = 25
                _pos_pix = pos_list[2*self.category_names.index(key):2*self.category_names.index(key)+2]
                # print("distance with ", key,": ",np.linalg.norm(_pos_pix-pix),_pos_pix,pix)
                if np.linalg.norm(_pos_pix-pix) < max_dis:
                    STEP_SIM = True
                    return STEP_SIM
        return False   
    def reward(self,observation,pick_success,action,env,BOUNDS,PIXEL_SIZE,xyz_to_pix):
        phase_penalty = 0.20
        time_penalty = 0.0
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
            dis_loss = - 0.1 if dis > 20 else 0
            # print("dis",dis,"pick_success",pick_success)
            # thre = 100
            # dis_loss = - (dis/thre,1)[dis>thre]**2*0.20
            reward += dis_loss
            reward -= 0.1 if not pick_success else 0
        if action[0] > 0.5 and env.last_pick_success:
            dis = np.linalg.norm(action[1:]-desired_action[1:])
            thre = 100
            dis_loss = - (dis/thre,1)[dis>thre]**2*0.20
            # dis_loss = - 0.1 if dis > 15 else 0
            reward += dis_loss  
        # print("desired_action :",desired_action,"phase_loss",phase_loss,"dis_reward",dis_loss)        
        done = (env.step_count >= env.max_steps)
        _reward,_done = self.get_reward(env)
        if _done:
            env.success = True
        else:
            reward -= time_penalty
            env.success = False
        reward += _reward
        done |= _done
        return reward,done
def sample_position(heatmap,MAX=False):
    # Flatten the heatmap
    if MAX:
        sampled_position = np.argmax(heatmap)
        sampled_position = np.unravel_index(sampled_position, heatmap.shape)
    else:
        flat = heatmap.flatten()
        # Create a cumulative distribution from the flattened heatmap
        cdf = np.cumsum(flat)
        # Sample a single value from the cumulative distribution
        random_value = np.random.rand()
        # Find the index in the CDF that corresponds to this random value
        sampled_index = np.searchsorted(cdf, random_value)
        # Convert the flat index back into 2D coordinates
        sampled_position = np.unravel_index(sampled_index, heatmap.shape)
    return sampled_position
class PutLetterontheBowlUnseen(PutLetterontheBowl):
    def __init__(self,colors,bounds):
        super().__init__(
            colors=colors,
            bounds=bounds,
        )
    def get_available_colors(self):
        return ["yellow","green","blue","cyan","orange","pink"]
class PutLetterontheBowlUnseenLetters(PutLetterontheBowl):
    def __init__(self,colors,bounds):
        config = {'pick':  ['Letter O', 'Letter T', 'Letter G'],
                  'place': ['yellow bowl', 'green bowl', 'blue bowl']}
        # letter T and Letter G unseen
        super().__init__(
            colors = colors,
            bounds = bounds,
            config = config
        )
    def get_available_colors(self):
        return ["yellow","green","blue","cyan","orange","pink"]
    
    def reset(self,env):
        # Load objects according to config.
        self.config = self.global_config
        self.config["pick"] = np.random.permutation(self.config["pick"]).tolist()
        # print("config",self.config)
        self.category_names = []
        for i in range(len(self.config["pick"])):
            self.category_names.append(self.config["pick"][i])
        for i in range(len(self.config["place"])):    
            if self.config["place"][i] not in self.category_names:
                self.category_names.append(self.config["place"][i])
        timeout = True
        obj_names = list(self.config["pick"]) + list(self.config["place"])
        while timeout:
            # print("init scene")
            pos,goals,timeout = self.plan_scene()

        self.goals = goals
        self.set_env(env,goals,obj_names,pos)

class PutblockonBowlSameColor(Task):
    """Put Blocks in Bowl with same color base class and task."""

    def __init__(self,colors,bounds, config = None):
        if config is None:
            config = {'pick':  ['yellow block', 'green block', 'blue block'],'place': ['yellow bowl', 'green bowl', 'blue bowl']}
        super().__init__(
            config = config,
            lang_template = "put the all letter on bowl of same color",
            goal_num = 0,
            colors = colors,
            bounds = bounds,
        )
        self.residual_region = 28//2
        self.category_names = []
        for i in range(len(self.global_config["pick"])):
            self.category_names.append(self.global_config["pick"][i])
        for i in range(len(self.global_config["place"])):    
            if self.global_config["place"][i] not in self.category_names:
                self.category_names.append(self.global_config["place"][i])
    def get_available_colors(self):
        return ["yellow","green","blue"]
    
    def reset(self,env,pos_dict = None):
        # Load objects according to config.
        self.config = copy.copy(self.global_config)
        # self.config["pick"] = np.random.permutation(self.config["pick"]).tolist()
        
        # self.config["place"] = np.random.permutation(self.config["place"]).tolist()
        # print("config",self.config)
        self.category_names = []
        for i in range(len(self.config["pick"])):
            self.category_names.append(self.config["pick"][i])
        for i in range(len(self.config["place"])):    
            if self.config["place"][i] not in self.category_names:
                self.category_names.append(self.config["place"][i])
        timeout = True
        obj_names = list(self.config["pick"]) + list(self.config["place"])
        if pos_dict is None:
            while timeout:
                # print("init scene")
                pos,goals,timeout = self.plan_scene()
        else:
            pos = []
            for obj_name in obj_names:
                pos.append(pos_dict[obj_name])
            # print("pos",pos)   
            goals = []

        self.goals = goals
        self.set_env(env,goals,obj_names,pos)

    def set_env(self,env,goals,obj_names,pos):
        obj_name_to_id = {}
        env.lang_goal = self.lang_template
        # print("obj_names",obj_names)
        for obj_name in obj_names:
            object_type = obj_name.split(" ")[1]
            rand_xyz = pos.pop(0)
            # if rand_xyz is np.array
            if isinstance(rand_xyz,np.ndarray):
                object_position = rand_xyz.squeeze()
            else:
                object_position = rand_xyz
            if object_type != "bowl" and object_type != "block":
                object_type = obj_name.split(" ")[2]
                object_color = self.colors[obj_name.split(" ")[0]]
                if object_type == "O":
                    shape = "environments/kitting/16.obj"
                elif object_type == "L":
                    shape = "environments/kitting/15.obj"
                elif object_type == "V":
                    shape = "environments/kitting/13.obj"
                elif object_type == "G":
                    shape = "environments/kitting/12.obj"
                elif object_type == "T":
                    shape = "environments/kitting/05.obj"
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
                theta = np.random.uniform(0, 2 * np.pi)
                object_orientation = pybullet.getQuaternionFromEuler([0, 0, theta])
                object_id = pybullet.loadURDF(fname, object_position,object_orientation, useFixedBase=0)
            elif object_type == "bowl":
                object_color = self.colors[obj_name.split(" ")[0]]
                object_position[2] = 0.003
                object_id = pybullet.loadURDF("environments/bowl/bowl.urdf", object_position, useFixedBase=1)
            elif object_type == "block":
                object_color = self.colors[obj_name.split(" ")[0]]
                object_shape = pybullet.createCollisionShape(pybullet.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02])
                object_visual = pybullet.createVisualShape(pybullet.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02])
                object_id = pybullet.createMultiBody(0.01, object_shape, object_visual, basePosition=object_position)
            pybullet.changeVisualShape(object_id, -1, rgbaColor=object_color)
            obj_name_to_id[obj_name] = object_id
        env.obj_name_to_id = obj_name_to_id
    def sample_goals(self):
        goals = []
        return goals
    def get_reward(self,env):
        dis = []
        for color in self.get_available_colors():
            bowls_name = color+" bowl"
            for _letter_name in self.config["pick"]:
                if _letter_name.split(" ")[0] == color:
                    letter_name = _letter_name

            bowl_pos = np.array(pybullet.getBasePositionAndOrientation(env.obj_name_to_id[bowls_name])[0])
            letter_pos = np.array(pybullet.getBasePositionAndOrientation(env.obj_name_to_id[letter_name])[0])
            dis.append(np.linalg.norm(bowl_pos-letter_pos))
        # print("dis",dis)
        if np.all(np.array(dis) < 0.07):
            return 1,True
        else:
            reward = -0.1*np.sum(np.array(dis) > 0.07)
            return reward,False 
    def reward(self,observation,pick_success,action,env,BOUNDS,PIXEL_SIZE,xyz_to_pix):
        phase_penalty = 0.20
        time_penalty = 0.0
        done = False
        reward = 0
        last_pos = env.obj_pos
        desired_primitive = env.last_pick_success

        y = env.last_pick_success
        cross_ent = y*np.log(action[0]+0.001)+(1-y)*np.log(1-action[0]+0.001)
        phase_loss = np.clip(-cross_ent*phase_penalty,0, phase_penalty)
        reward -= phase_loss
        done = (env.step_count >= env.max_steps)


        _reward,_done = self.get_reward(env)
        reward += _reward
        done |= _done
        if _done:
            env.success = True
        else:
            env.success = False
        
        return reward,done
    def exploration_action(self,env,deterministic=False):
        # print("get_expert_demonstration")
        # print(self.observation)
        obs = env.observation
        object_in_hand = env.last_pick_success
        obj_pos = env.pos_list
        goals = self.goals
        obj_num = len(self.category_names)
        picked_obj = None

        if object_in_hand:
            prim = 1
            # objec_in_hand 
            picked_obj = None
            for obj in self.config['pick']:
                if env.obj_pos[obj][2] >= 0.2:
                    color = obj.split(" ")[0]
            placed_obj = color+" bowl"
            placed_index = 3+self.config["place"].index(placed_obj)
            one_hot_idx = [1 if i== placed_index else 0  for i in range(obj_num)]
            res = [0,0]
            
        else:
            prim = 0
            colors = self.get_available_colors()
            colors = np.random.permutation(colors).tolist()
            for color in colors:
                bowls_name = color+" bowl"
                block_name = color+" block"
                block_index = self.config["pick"].index(color+" block")
                bowl_index = 3+self.config["place"].index(color+" bowl")
                block_pos = np.array(obj_pos[2*block_index:2*block_index+2])
                bowl_pos = np.array(obj_pos[2*bowl_index:2*bowl_index+2])
                # print(bowls_name,bowl_pos,block_name,block_pos)
                if np.linalg.norm(block_pos-bowl_pos) > 10:
                    idx = block_index
                    res = [0,0]
                    break
            one_hot_idx = [1 if i== idx else 0  for i in range(obj_num)]
        

        if env.scale_action:
            res = np.array(res)/env.residual_region
        desired_action = [prim]+ one_hot_idx + res.tolist()            

        return np.array(desired_action) 
            
        

        



