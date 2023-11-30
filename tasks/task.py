import numpy as np
from environments.environment import BOUNDS,PIXEL_SIZE,COLORS
import pybullet
class Task():
    def __init__(self,
                 config, # pick&place objects
                 lang_template,
                 goal_num, # number of goals
                 colors = COLORS,
                 bounds = BOUNDS,
                 dis_eps = 0.15,
                 ):
        self.colors = colors
        self.bounds = bounds
        self.global_config = config
        
        self.lang_template = lang_template
        self.goal_num = goal_num
        self.max_steps = 10
        self.dis_eps = dis_eps
        self.pos_eps = 0.05
    
    def reset(self,env):
        # Load objects according to config.
        self.config = self.global_config
        self.category_names = []
        for i in range(len(self.config["pick"])):
            self.category_names.append(self.config["pick"][i])
        for i in range(len(self.config["place"])):    
            if self.config["place"][i] not in self.category_names:
                self.category_names.append(self.config["place"][i])
        timeout = True
        obj_names = list(self.config["pick"]) + list(self.config["place"])
        while timeout:
            print("init scene")
            pos,goals,timeout = self.plan_scene()

        self.goals = goals
        self.set_env(env,goals,obj_names,pos)
        

    def plan_scene(self):
        time = 0
        obj_names = list(self.config["pick"]) + list(self.config["place"])
        obj_xyz = np.zeros((0, 3))
        pos = []
        for obj_name in obj_names:               
            # Get random position 15cm+ from other objects.
            while True:
                time += 1
                if time > 100:
                    timeout = True
                    return None,None,timeout
                rand_x = np.random.uniform(self.bounds[0, 0] + 0.1, self.bounds[0, 1] - 0.1)
                rand_y = np.random.uniform(self.bounds[1, 0] + 0.1, self.bounds[1, 1] - 0.1)
                rand_xyz = np.float32([rand_x, rand_y, 0.03]).reshape(1, 3)
                if len(obj_xyz) == 0:
                    obj_xyz = np.concatenate((obj_xyz, rand_xyz), axis=0)
                    break
                else:
                    nn_dist = np.min(np.linalg.norm(obj_xyz - rand_xyz, axis=1)).squeeze()
                    if nn_dist > self.dis_eps:
                        obj_xyz = np.concatenate((obj_xyz, rand_xyz), axis=0)
                        break
            pos.append(rand_xyz)
                
        # determine pick&place goal
        goals = self.sample_goals()
        
        timeout = False
        return pos,goals,timeout
    
    def set_env(self,env,goals,obj_names,pos):
        raise NotImplementedError

    def sample_goals(self):
        raise NotImplementedError

    def p_neglect(self,pix,obj_name_to_id,pos_list):
        for key in obj_name_to_id.keys():
            if key in self.config["pick"] : 
                max_dis = 12
                _pos_pix = pos_list[2*self.category_names.index(key):2*self.category_names.index(key)+2]
                # print("distance with ", key,": ",np.linalg.norm(_pos_pix-pix),_pos_pix,pix)
                if np.linalg.norm(_pos_pix-pix) < max_dis:
                    STEP_SIM = True
                    return STEP_SIM
        return False
class PutBlockInBowl(Task):
    """Put Blocks in Bowl base class and task."""

    def __init__(self,colors,bounds):
        super().__init__(
            config = {'pick':  ['yellow block', 'green block', 'blue block'],'place': ['yellow bowl', 'green bowl', 'blue bowl']},
            lang_template = "put the {pick} on a {place}",
            goal_num = 2,
            colors = colors,
            bounds = bounds,
        )
    def set_env(self,env,goals,obj_names,pos):
        pick_idx,place_idx = goals
        obj_name_to_id = {}
        env.lang_goal = self.lang_template.format(pick=self.config["pick"][pick_idx], place=self.config["place"][place_idx])
        for obj_name in obj_names:
            object_color = self.colors[obj_name.split(" ")[0]]
            object_type = obj_name.split(" ")[1]
            rand_xyz = pos.pop(0)
            object_position = rand_xyz.squeeze()
            if object_type == "block":
                object_shape = pybullet.createCollisionShape(pybullet.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02])
                object_visual = pybullet.createVisualShape(pybullet.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02])
                object_id = pybullet.createMultiBody(0.01, object_shape, object_visual, basePosition=object_position)
            elif object_type == "bowl":
                object_position[2] = 0
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
        if object_in_hand:
            picked_obj = 0
            for obj in self.config['pick']:
                if env.obj_pos[obj][2] >= 0.2:
                    picked_obj = obj
                    print("picked_obj",picked_obj)
            
            if picked_obj == self.config['pick'][goals[0]]:
                place_obj = goals[1]+len(self.config['pick'])
                angle = np.random.uniform(0,2*np.pi)
                r = np.random.uniform(0,2)
                res = np.array([np.cos(angle),np.sin(angle)])*r
                # res = np.array([0,0])
            else:
                place_obj = np.random.randint(0,obj_num)
                res = np.random.randint(0,env.image_size[0]/4,(2,))

            if env.one_hot_action:
                one_hot_idx = [1 if i== place_obj else 0  for i in range(obj_num+1)]
                idx = one_hot_idx
            else:
                idx = [place_obj]

            if env.scale_action:
                res = np.array(res)/env.image_size[0]
            desired_action = [1]+ idx + res.tolist()
            

        else:
            picked_obj = goals[0]
            if env.one_hot_action:
                one_hot_idx = [1 if i== picked_obj else 0  for i in range(obj_num+1)]
                idx = one_hot_idx
            else:
                idx = [picked_obj]
            angle = np.random.uniform(0,2*np.pi)
            r = 0
            res = np.array([np.cos(angle),np.sin(angle)])*r
            if env.scale_action:
                res = np.array(res)/env.image_size[0]
            desired_action = [0]+ idx + res.tolist()

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


class PickBlock():
    def __init__(self,colors,bounds):
        super().__init__()
        self.max_steps = 10
        self.pos_eps = 0.05
        self.lang_template = "pick the {pick}"
        self.colors = colors
        self.bounds = bounds
        self.goal_num = 1
        self.category_names = ['yellow block', 'green block', 'blue block']
        self.config =  {'pick':  ['yellow block', 'green block', 'blue block'],'place':[]}
    def reset(self,env):
        # Load objects according to config.
        timeout = True
        obj_name_to_id = {}
        obj_names = list(self.config["pick"])
        while timeout:
            print("init scene")
            pos,goals,timeout = self.plan_scene()
        
        self.goals = goals
        pick_idx = goals[0]
        env.lang_goal = self.lang_template.format(pick=self.config["pick"][pick_idx])
        for obj_name in obj_names:
            object_color = self.colors[obj_name.split(" ")[0]]
            object_type = obj_name.split(" ")[1]
            rand_xyz = pos.pop(0)
            object_position = rand_xyz.squeeze()
            if object_type == "block":
                object_shape = pybullet.createCollisionShape(pybullet.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02])
                object_visual = pybullet.createVisualShape(pybullet.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02])
                object_id = pybullet.createMultiBody(0.01, object_shape, object_visual, basePosition=object_position)
            
            pybullet.changeVisualShape(object_id, -1, rgbaColor=object_color)
            obj_name_to_id[obj_name] = object_id
        env.obj_name_to_id = obj_name_to_id

    def p_neglect(self,pix,obj_name_to_id,pos_list):
        for key in obj_name_to_id.keys():
            if key in self.config["pick"] : 
                max_dis = 15
            _pos_pix = pos_list[2*self.category_names.index(key):2*self.category_names.index(key)+2]
            # print("distance with ", key,": ",np.linalg.norm(_pos_pix-pix),_pos_pix,pix)
            if np.linalg.norm(_pos_pix-pix) < max_dis:
                STEP_SIM = True
                return STEP_SIM
        return False
    
        
    def plan_scene(self):
        time = 0
        obj_names = list(self.config["pick"])
        obj_xyz = np.zeros((0, 3))
        pos = []
        for obj_name in obj_names:
            if ("block" in obj_name) :
                # Get random position 15cm+ from other objects.
                while True:
                    time += 1
                    if time > 100:
                        timeout = True
                        return None,None,timeout
                    rand_x = np.random.uniform(self.bounds[0, 0] + 0.1, self.bounds[0, 1] - 0.1)
                    rand_y = np.random.uniform(self.bounds[1, 0] + 0.1, self.bounds[1, 1] - 0.1)
                    rand_xyz = np.float32([rand_x, rand_y, 0.03]).reshape(1, 3)
                    if len(obj_xyz) == 0:
                        obj_xyz = np.concatenate((obj_xyz, rand_xyz), axis=0)
                        break
                    else:
                        nn_dist = np.min(np.linalg.norm(obj_xyz - rand_xyz, axis=1)).squeeze()
                        if nn_dist > 0.15:
                            obj_xyz = np.concatenate((obj_xyz, rand_xyz), axis=0)
                            break
                pos.append(rand_xyz)
                
        # determine pick&place goal 
        pick_idx =np.random.choice(len(self.config["pick"]))
        goals = [pick_idx]
        # print("goals:",self.config["pick"][pick_idx])
        timeout = False
        return pos,goals,timeout
    
        # env.lang_goal = "put all blocks in different corners" 
    def get_reward(self,env):
        pick_idx = self.goals[0]
        pick_pos = pybullet.getBasePositionAndOrientation(env.obj_name_to_id[self.config["pick"][pick_idx]])[0]

        height = pick_pos[2]
        if height > 0.2:
            return 1,True
        else:
            return 0,False
    def reward(self,observation,action,env,BOUNDS,PIXEL_SIZE,xyz_to_pix):
        phase_penalty = 0.25
        done = False
        reward = 0
        last_pos = env.obj_pos
        desired_primitive = env.last_pick_success
        if env.last_pick_success:
            desired_position = [0,0]
            print("desired_position",desired_position)
            desired_position = xyz_to_pix(desired_position,BOUNDS,PIXEL_SIZE)
            desired_action = [desired_primitive]+desired_position[:2]
        else:
            desired_position = last_pos[self.config['pick'][self.goals[0]]]
            desired_position = xyz_to_pix(desired_position,BOUNDS,PIXEL_SIZE)
            desired_action = [desired_primitive]+desired_position[:2]
        y = env.last_pick_success
        print("desired_action :",desired_action)
        cross_ent = y*np.log(action[0]+0.001)+(1-y)*np.log(1-action[0]+0.001)
        phase_loss = -cross_ent*phase_penalty
        phase_loss = np.clip(phase_loss,0, phase_penalty)
        reward -= phase_loss
        print("phase_loss",phase_loss)
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
            # dis_loss = - (dis/thre,1)[dis>thre]**2*0.25
            reward += dis_loss  
        print("dis_reward",dis_loss)        
        done = (env.step_count >= env.max_steps)
        _reward,_done = self.get_reward(env)
        if _done:
            env.success = True
        else:
            env.success = False
        reward += _reward
        done |= _done
        return reward,done
class PutBowlInBowl():
    """Put Blocks in Bowl base class and task."""

    def __init__(self,colors,bounds):
        super().__init__()
        self.max_steps = 10
        self.pos_eps = 0.05
        self.lang_template = "put the {pick} in a {place}"
        self.colors = colors
        self.bounds = bounds
        self.category_names = ['yellow bowl', 'green bowl', 'blue bowl',]
        self.config =  {'pick':  ['yellow bowl', 'green bowl', 'blue bowl',],
          'place': ['yellow bowl', 'green bowl', 'blue bowl']}
        self.goal_num = 2
                            #    "blue block in green bowl",
                            #    "green block in blue bowl", 
                            #    "yellow block in blue bowl", 
                            #    "blue block in yellow bowl",
                            #    "yellow block in green bowl",
                            #    "green block in yellow bowl"]

    def reset(self,env):
        # Load objects according to config.
        
        timeout = True
        obj_name_to_id = {}
        obj_names = list(self.config["pick"]) + list(self.config["place"])
        obj_names = set(obj_names)
        while timeout:
            print("init scene")
            pos,goals,timeout = self.plan_scene()
        
        self.goals = goals
        pick_idx,place_idx = goals
        env.lang_goal = self.lang_template.format(pick=self.config["pick"][pick_idx], place=self.config["place"][place_idx])
        for obj_name in obj_names:
            # print("obj_name:",obj_name)
            object_color = self.colors[obj_name.split(" ")[0]]
            object_type = obj_name.split(" ")[1]
            rand_xyz = pos.pop(0)
            object_position = rand_xyz.squeeze()
            if object_type == "block":
                object_shape = pybullet.createCollisionShape(pybullet.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02])
                object_visual = pybullet.createVisualShape(pybullet.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02])
                object_id = pybullet.createMultiBody(0.01, object_shape, object_visual, basePosition=object_position)
            elif object_type == "bowl":
                object_position[2] = 0
                object_id = pybullet.loadURDF("environments/bowl/bowl.urdf", object_position, useFixedBase=0)
            pybullet.changeVisualShape(object_id, -1, rgbaColor=object_color)
            obj_name_to_id[obj_name] = object_id
        env.obj_name_to_id = obj_name_to_id

    def p_neglect(self,pix,obj_name_to_id,pos_list):
        for key in obj_name_to_id.keys():
            if key in self.config["pick"] or key in self.config["place"]: 
                max_dis = 25
            _pos_pix = pos_list[2*self.category_names.index(key):2*self.category_names.index(key)+2]
            print("distance with ", key,": ",np.linalg.norm(_pos_pix-pix),_pos_pix,pix)
            if np.linalg.norm(_pos_pix-pix) < max_dis:
                STEP_SIM = True
                return STEP_SIM
        return False

        
    def plan_scene(self):
        time = 0
        obj_names = list(self.config["pick"]) + list(self.config["place"])
        obj_names = set(obj_names)
        obj_xyz = np.zeros((0, 3))
        pos = []
        for obj_name in obj_names:
            if ("block" in obj_name) or ("bowl" in obj_name):
                # Get random position 15cm+ from other objects.
                while True:
                    time += 1
                    if time > 100:
                        timeout = True
                        return None,None,timeout
                    rand_x = np.random.uniform(self.bounds[0, 0] + 0.1, self.bounds[0, 1] - 0.1)
                    rand_y = np.random.uniform(self.bounds[1, 0] + 0.1, self.bounds[1, 1] - 0.1)
                    rand_xyz = np.float32([rand_x, rand_y, 0.03]).reshape(1, 3)
                    if len(obj_xyz) == 0:
                        obj_xyz = np.concatenate((obj_xyz, rand_xyz), axis=0)
                        break
                    else:
                        nn_dist = np.min(np.linalg.norm(obj_xyz - rand_xyz, axis=1)).squeeze()
                        if nn_dist > 0.15:
                            obj_xyz = np.concatenate((obj_xyz, rand_xyz), axis=0)
                            break
                pos.append(rand_xyz)
                
        # # determine pick&place goal 
        # pick_idx =np.random.choice(len(self.config["pick"]))
        # place_idx = np.random.choice(len(self.config["place"]))
        goals = np.random.choice(len(self.config["pick"]),2,replace=False)
        # print("goals:",self.config["pick"][pick_idx])
        timeout = False
        return pos,goals,timeout
        # env.lang_goal = "put all blocks in different corners" 
    def get_reward(self,env):
        pick_idx,place_idx = self.goals
        pick_pos = pybullet.getBasePositionAndOrientation(env.obj_name_to_id[self.config["pick"][pick_idx]])[0]
        place_pos = pybullet.getBasePositionAndOrientation(env.obj_name_to_id[self.config["place"][place_idx]])[0]
        # print("pick_pos:",pick_pos,"place_pos:",place_pos)
        # dis = np.linalg.norm(np.array(pick_pos)[:2] - np.array(place_pos)[:2])
        # print("dis:",dis)
        if np.linalg.norm(np.array(pick_pos)[:2] - np.array(place_pos)[:2]) < self.pos_eps:
        
            return 1,True
        else:
            return 0,False
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
        # print("desired_action :",desired_action)
        cross_ent = y*np.log(action[0]+0.001)+(1-y)*np.log(1-action[0]+0.001)
        print("cross_ent",cross_ent)
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
        print("phase_loss: ",phase_loss," dis_reward: ",dis_loss)        
        done = (env.step_count >= env.max_steps)
        _reward,_done = self.get_reward(env)
        if _done:
            env.success = True
        else:
            env.success = False
        reward += _reward
        done |= _done
        return reward,done
    def exploration_action(self,env):
        print("get_expert_demonstration")
        # print(self.observation)
        object_in_hand = env.last_pick_success
        obj_pos = env.pos_list
        goals = self.goals
        obj_num = len(self.category_names)
        print("obj_pos",obj_pos)
        print("goals",goals)
        if object_in_hand:
            picked_obj = 0
            for obj in self.config['pick']:
                if env.obj_pos[obj][2] >= 0.2:
                    picked_obj = obj
                    print("picked_obj",picked_obj)
            
            if picked_obj == self.config['pick'][goals[0]]:
                place_obj = goals[1]
                if env.ee_type == "gripper":
                    angle = np.random.uniform(0,2*np.pi)
                    r = np.random.uniform(0,20)
                    res = np.array([np.cos(angle),np.sin(angle)])*r
                elif env.ee_type == "suction":
                    res = np.random.normal([0,0],[5,5])
                # res = np.array([0,0])
            else:
                place_obj = np.random.randint(0,obj_num)
                res = np.random.randint(0,env.image_size[0]/4,(2,))

            if env.one_hot_action:
                one_hot_idx = [1 if i== place_obj else 0  for i in range(obj_num)]
                idx = one_hot_idx
            else:
                idx = [place_obj]

            if env.scale_action:
                res = np.array(res)/env.image_size[0]
            desired_action = [1]+ idx + res.tolist()
            

        else:
            picked_obj = goals[0]
            if env.one_hot_action:
                one_hot_idx = [1 if i== picked_obj else 0  for i in range(obj_num)]
                idx = one_hot_idx
            else:
                idx = [picked_obj]
            angle = np.random.uniform(0,2*np.pi)
            r = 18
            res = np.array([np.cos(angle),np.sin(angle)])*r
            if env.scale_action:
                res = np.array(res)/env.image_size[0]
            desired_action = [0]+ idx + res.tolist()

        print("desired_action",desired_action)
        return np.array(desired_action)
class PickBowl():
    def __init__(self,colors,bounds):
        super().__init__()
        self.max_steps = 10
        self.pos_eps = 0.05
        self.lang_template = "pick the {pick}"
        self.colors = colors
        self.bounds = bounds
        self.goal_num = 1
        self.category_names = ['yellow bowl', 'green bowl', 'blue bowl',]
        self.config =  {'pick':  ['yellow bowl', 'green bowl', 'blue bowl',],'place':[]}
    def reset(self,env):
        # Load objects according to config.
        timeout = True
        obj_name_to_id = {}
        obj_names = list(self.config["pick"])
        while timeout:
            print("init scene")
            pos,goals,timeout = self.plan_scene()
        
        self.goals = goals
        pick_idx = goals[0]
        env.lang_goal = self.lang_template.format(pick=self.config["pick"][pick_idx])
        for obj_name in obj_names:
            object_color = self.colors[obj_name.split(" ")[0]]
            object_type = obj_name.split(" ")[1]
            rand_xyz = pos.pop(0)
            object_position = rand_xyz.squeeze()
            if object_type == "bowl":
                object_position[2] = 0
                object_id = pybullet.loadURDF("environments/bowl/bowl.urdf", object_position, useFixedBase=0)
            
            pybullet.changeVisualShape(object_id, -1, rgbaColor=object_color)
            obj_name_to_id[obj_name] = object_id
        env.obj_name_to_id = obj_name_to_id

    def p_neglect(self,pix,obj_name_to_id,pos_list):
        for key in obj_name_to_id.keys():
            if key in self.config["pick"] : 
                max_dis = 30
            _pos_pix = pos_list[2*self.category_names.index(key):2*self.category_names.index(key)+2]
            print("distance with ", key,": ",np.linalg.norm(_pos_pix-pix),_pos_pix,pix)
            if np.linalg.norm(_pos_pix-pix) < max_dis:
                STEP_SIM = True
                return STEP_SIM
        return False
    
        
    def plan_scene(self):
        time = 0
        obj_names = list(self.config["pick"])
        obj_xyz = np.zeros((0, 3))
        pos = []
        for obj_name in obj_names:
            if ("bowl" in obj_name) :
                # Get random position 15cm+ from other objects.
                while True:
                    time += 1
                    if time > 100:
                        timeout = True
                        return None,None,timeout
                    rand_x = np.random.uniform(self.bounds[0, 0] + 0.1, self.bounds[0, 1] - 0.1)
                    rand_y = np.random.uniform(self.bounds[1, 0] + 0.1, self.bounds[1, 1] - 0.1)
                    rand_xyz = np.float32([rand_x, rand_y, 0.03]).reshape(1, 3)
                    if len(obj_xyz) == 0:
                        obj_xyz = np.concatenate((obj_xyz, rand_xyz), axis=0)
                        break
                    else:
                        nn_dist = np.min(np.linalg.norm(obj_xyz - rand_xyz, axis=1)).squeeze()
                        if nn_dist > 0.15:
                            obj_xyz = np.concatenate((obj_xyz, rand_xyz), axis=0)
                            break
                pos.append(rand_xyz)
                
        # determine pick&place goal 
        pick_idx =np.random.choice(len(self.config["pick"]))
        goals = [pick_idx]
        # print("goals:",self.config["pick"][pick_idx])
        timeout = False
        return pos,goals,timeout
    
        # env.lang_goal = "put all blocks in different corners" 
    def get_reward(self,env):
        pick_idx = self.goals[0]
        pick_pos = pybullet.getBasePositionAndOrientation(env.obj_name_to_id[self.config["pick"][pick_idx]])[0]

        height = pick_pos[2]
        if height > 0.2:
            return 1,True
        else:
            return 0,False
    def reward(self,observation,action,env,BOUNDS,PIXEL_SIZE,xyz_to_pix):
        phase_penalty = 0.25
        done = False
        reward = 0
        last_pos = env.obj_pos
        desired_primitive = env.last_pick_success
        if env.last_pick_success:
            desired_position = [0,0]
            print("desired_position",desired_position)
            desired_position = xyz_to_pix(desired_position,BOUNDS,PIXEL_SIZE)
            desired_action = [desired_primitive]+desired_position[:2]
        else:
            desired_position = last_pos[self.config['pick'][self.goals[0]]]
            desired_position = xyz_to_pix(desired_position,BOUNDS,PIXEL_SIZE)
            desired_action = [desired_primitive]+desired_position[:2]
        y = env.last_pick_success
        print("desired_action :",desired_action)
        cross_ent = y*np.log(action[0]+0.001)+(1-y)*np.log(1-action[0]+0.001)
        phase_loss = -cross_ent*phase_penalty
        phase_loss = np.clip(phase_loss,0, phase_penalty)
        reward -= phase_loss
        print("phase_loss",phase_loss)
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
            # dis_loss = - (dis/thre,1)[dis>thre]**2*0.25
            reward += dis_loss  
        print("dis_reward",dis_loss)        
        done = (env.step_count >= env.max_steps)
        _reward,_done = self.get_reward(env)
        if _done:
            env.success = True
        else:
            env.success = False
        reward += _reward
        done |= _done
        return reward,done
    def exploration_action(self,env):
        print("get_expert_demonstration")
        # print(self.observation)
        object_in_hand = env.last_pick_success
        obj_pos = env.pos_list
        goals = self.goals
        obj_num = len(self.category_names)
        print("obj_pos",obj_pos)
        print("goals",goals)
        if object_in_hand:
            picked_obj = 0
            desired_action = env.action_space.sample()
            desired_action[0] = 1        

        else:
            picked_obj = goals[0]
            if env.one_hot_action:
                one_hot_idx = [1 if i== picked_obj else 0  for i in range(obj_num)]
                idx = one_hot_idx
            else:
                idx = [picked_obj]
            angle = np.random.uniform(0,2*np.pi)
            r = 18
            res = np.array([np.cos(angle),np.sin(angle)])*r
            if env.scale_action:
                res = np.array(res)/env.image_size[0]
            desired_action = [0]+ idx + res.tolist()

        print("desired_action",desired_action)
        return np.array(desired_action)
        

        