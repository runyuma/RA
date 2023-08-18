import numpy as np
import random
import pybullet
class PutBlockInBowl():
    """Put Blocks in Bowl base class and task."""

    def __init__(self,colors,bounds):
        super().__init__()
        self.max_steps = 10
        self.pos_eps = 0.05
        self.lang_template = "put the {pick} in a {place}"
        self.colors = colors
        self.bounds = bounds
        self.category_names = ['yellow block', 'green block', 'blue block', 'yellow bowl', 'green bowl', 'blue bowl',]
                            #    "blue block in green bowl",
                            #    "green block in blue bowl", 
                            #    "yellow block in blue bowl", 
                            #    "blue block in yellow bowl",
                            #    "yellow block in green bowl",
                            #    "green block in yellow bowl"]

    def reset(self,env):
        # Load objects according to config.
        self.config =  {'pick':  ['yellow block', 'green block', 'blue block'],
          'place': ['yellow bowl', 'green bowl', 'blue bowl']}
        env.obj_name_to_id = {}
        obj_names = list(self.config["pick"]) + list(self.config["place"])
        obj_xyz = np.zeros((0, 3))
        for obj_name in obj_names:
            if ("block" in obj_name) or ("bowl" in obj_name):
                # Get random position 15cm+ from other objects.
                while True:
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

                object_color = self.colors[obj_name.split(" ")[0]]
                object_type = obj_name.split(" ")[1]
                object_position = rand_xyz.squeeze()
                if object_type == "block":
                    object_shape = pybullet.createCollisionShape(pybullet.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02])
                    object_visual = pybullet.createVisualShape(pybullet.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02])
                    object_id = pybullet.createMultiBody(0.01, object_shape, object_visual, basePosition=object_position)
                elif object_type == "bowl":
                    object_position[2] = 0
                    object_id = pybullet.loadURDF("environments/bowl/bowl.urdf", object_position, useFixedBase=1)
                pybullet.changeVisualShape(object_id, -1, rgbaColor=object_color)
                env.obj_name_to_id[obj_name] = object_id
        # determine pick&place goal 
        pick_idx =np.random.choice(len(self.config["pick"]))
        place_idx = np.random.choice(len(self.config["place"]))
        self.goals = [pick_idx,place_idx]
        # print("goals:",self.config["pick"][pick_idx])
        # env.lang_goal = self.lang_template.format(pick=self.config["pick"][pick_idx], place=self.config["place"][place_idx])
        env.lang_goal = "put all blocks in different corners"
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
        

        