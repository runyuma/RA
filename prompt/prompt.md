# Prompt for this project
## exploration prompt
1. planning prompt
```
write a fuction for high-level planning of robot pick&place, given the language input and objects, input of this function is a list of past experience of robot. Output is the desired behavior of this robot, for example: robot.pick(pick_obj) robot.place(picked_obj,place_location).
You should know: you can run robot.pick() if and only if the robot have not picked up an object successfully on last step. You can run robot.place(picked_obj,place_location) if and only if picked_obj have been picked up successfully on last step.
Example:
language command: put all blocks in different bowls
def function(previous_actions):
    available_blocks = [block for block in blocks if not block.in_bowl()]
    avaiable_bowls = [bowl for bowl in bowls if not bowl.empty()]
    if previous_action[-1].is_pick and previous_action[-1].is_success:
        place_bowl = np.random.choice(avaiable_bowls)
        return robot.place(previous_action[-1].picked_obj,place_bowl)
    else:
        pick_block = np.random.choice(available_blocks)
        return robot.pick(pick_block)
Please write a function with language input: put block{A} on bowl{B}
```
language command: put block{A} on bowl{B} result from gpt-4
```python
# put {block_A} on {bowl_B}
def PickBlockOnBowl(previous_actions, block_A, bowl_B):
    if previous_actions[-1].is_pick \
        and previous_actions[-1].is_success \
        and previous_actions[-1].picked_obj == block_A:
        return robot.place(block_A, bowl_B)
    else:
        return robot.pick(block_A)
```
language command:stack all blocks together result from gpt-4
```python
# stack all blocks together
def stack_blocks(previous_actions):
    available_blocks = [block for block in blocks if not block.is_stacked()]
    if len(available_blocks) == 0:
        return None  # All blocks are already stacked
    if previous_actions[-1].is_pick and previous_actions[-1].is_success:
        # If the last action was a successful pick, place the block on the stack
        stack_location = blocks[0].location
        return robot.place(previous_actions[-1].picked_obj, stack_location)
    else:
        # If the last action was not a successful pick, try to pick up a block
        pick_block = np.random.choice(available_blocks)
        return robot.pick(pick_block)
```
language command:put all objects in a line
You should automated regress this line(k,b) according to the position of objects in [224,224] 2D space.
```python
import numpy as np
from sklearn.linear_model import LinearRegression

def function(previous_actions, objects):
    # Get the positions of the objects
    positions = np.array([obj.position for obj in objects])
    
    # Fit a linear regression model to the positions
    model = LinearRegression().fit(positions[:,0].reshape(-1,1), positions[:,1])
    
    # Get the slope and intercept of the line
    k = model.coef_[0]
    b = model.intercept_
    
    # Get the objects that are not in line
    not_in_line = [obj for obj in objects if abs(obj.position[1] - (k * obj.position[0] + b)) > 1e-6]
    
    # If the robot has picked up an object in the last step, place it
    if previous_actions[-1].is_pick and previous_actions[-1].is_success:
    #     return robot.place(previous_actions[-1].picked_obj, not_in_line[0])
    # else:
    #     # Otherwise, pick up an object that is not in line
    #     return robot.pick(not_in_line[0])
import numpy as np
from sklearn.linear_model import LinearRegression

def function(previous_actions, objects):
    # Check if the robot has picked up an object in the last step
    if previous_actions[-1].is_pick and previous_actions[-1].is_success:
        # If the robot has picked up an object, place it on the line
        # Calculate the position on the line for the object
        x_position = previous_actions[-1].picked_obj.position[0]
        y_position = k * x_position + b
        return robot.place(previous_actions[-1].picked_obj, (x_position, y_position))
    else:
        # If the robot has not picked up an object, pick up an object
        available_objects = [obj for obj in objects if not obj.in_line()]
        if available_objects:
            pick_object = np.random.choice(available_objects)
            return robot.pick(pick_object)
        else:
            return None  # No more objects to pick up

# Get the positions of the objects
object_positions = np.array([obj.position for obj in objects])

# Fit a linear regression model to the object positions
reg = LinearRegression().fit(object_positions[:, 0].reshape(-1, 1), object_positions[:, 1])

# Get the slope (k) and intercept (b) of the line
k = reg.coef_[0]
b = reg.intercept_
```


2. grab prompt
    ```
    You need to write a function to sample the pick position in 2D for an object given the object's size. Given the bounding box, you need to give high value to the position that is easy to pick by a parallel gripper. You also need to make sure that every position in the bounding box have a chance to be sampled.  
    For example:
    object: block
    def explore_probability(h,w)
        # h: height of the object
        # w: width of the object
        # return: a 2D array with the same size of the bounding box, each element is the probability of sampling this position
        center = [h/2,w/2]
        probability = np.zeros((h,w))
        for i in range(h):
            for j in range(w):
                probability[i,j] = 1 - distance([i,j],center)/(h/2+w/2)
        probability = probability/np.sum(probability)
        return probability
    Now you need write a function for:
    obj={obj}.You should consider the shape of given object.
    ```

    object: bowl from **gpt-4**
    
    ```python
    # pick position for bowl
    import numpy as np
    def explore_probability(h, w):
        # h: height of the bowl
        # w: width of the bowl
        # return: a 2D array with the same size of the bounding box, each element is the probability of sampling this position

        # Initialize the probability array
        probability = np.zeros((h, w))
        # Calculate the center of the bowl
        center = [h/2, w/2]
        # Calculate the radius of the bowl
        radius = min(h, w) / 2
        # Iterate over each position in the bounding box
        for i in range(h):
            for j in range(w):
                # Calculate the distance from the current position to the center of the bowl
                distance_to_center = np.sqrt((i - center[0])**2 + (j - center[1])**2)
                # If the position is near the rim of the bowl, assign a higher probability
                if radius - 0.1*radius <= distance_to_center <= radius + 0.1*radius:
                    probability[i, j] = 1 - abs(distance_to_center - radius) / (0.1*radius)
                else:
                    probability[i, j] = 0
        # Normalize the probabilities so they sum to 1
        probability = probability / np.sum(probability)
        return probability
    ```
    ### orientation ?
    This method only works for objects that are symmetric about the center of the bounding box. For objects that are not symmetric, the orientation is not in action space/ observation space and it is very hard to get in VLM.
3. conclude prompt

## prompt for reward
1. intermediate prompt
```
Language command: {lang_command}
You should design a function based on the language, action of last step and the current state of the environment
# for template:
def reward_function(previous_actions, current_state):
    # action.action_type: pick/place
    # action.position: position of the action, low = [0,0],high = [224,224]
    # current_state.object_inhand: True if the robot has picked up an object
    # current_state.object_position: {object_i: [x,y]}, low = [0,0],high = [224,224]

    # reward can be a combination of 2 cost: phase cost and dist cost
    # phase_cost(action.action_type,detected_action_type) = - 0.1 if action.action_type != detected_action_type else 0
    # dist_cost(action.position,object_position) = - 0.1 * distance(action.position-object_position)

    return reward
# for example:
pick up the all block in bowls
def reward_function(actions, current_state):
    if not current_state.object_inhand:
        closest_block = argmin(distance([block for block in blocks],action.position))
        dist_cost = dist_cost(action.position,closest_block.position)
        phase_cost = phase_cost(action.action_type,'pick')
        return dist_cost + phase_cost
    else:
        closest_bowl = argmin(distance([bowl for bowl in bowls],action.position))
        dist_cost = dist_cost(action.position,closest_bowl.position)
        phase_cost = phase_cost(action.action_type,'place')
        return dist_cost + phase_cost
Now please finish a reward function for Language command: put {block A} on {bowl B}
```
put {block A} on {bowl B}
```
def reward_function(actions, current_state):
    block_A = actions[0].block
    bowl_B = actions[0].bowl

    if not current_state.object_inhand:
        block_A_position = current_state.object_position[block_A]
        dist_cost = -0.1 * distance(actions[0].position - block_A_position)
        phase_cost = -0.1 if actions[0].action_type != 'pick' else 0
        return dist_cost + phase_cost
    else:
        bowl_B_position = current_state.object_position[bowl_B]
        dist_cost = -0.1 * distance(actions[0].position - bowl_B_position)
        phase_cost = -0.1 if actions[0].action_type != 'place' else 0
        return dist_cost + phase_cost
```
stack all blocks together

```
gpt-3
def reward_function(actions, current_state):
    if not current_state.object_inhand:
        closest_block = argmin(distance([block for block in blocks], action.position))
        dist_cost = dist_cost(action.position, closest_block.position)
        phase_cost = phase_cost(action.action_type, 'pick')
        return dist_cost + phase_cost
    else:
        closest_block = argmin(distance([block for block in blocks], action.position))
        dist_cost = dist_cost(action.position, closest_block.position)
        phase_cost = phase_cost(action.action_type, 'place')
        return dist_cost + phase_cost
gpt-4        
def reward_function(actions, current_state):
    if not current_state.object_inhand:
        # If the robot does not have a block in hand, it should pick one up
        # The closest block is chosen
        closest_block = argmin(distance([block for block in blocks],action.position))
        dist_cost = dist_cost(action.position,closest_block.position)
        phase_cost = phase_cost(action.action_type,'pick')
        return dist_cost + phase_cost
    else:
        # If the robot has a block in hand, it should place it on top of the stack
        # The position of the top of the stack is chosen
        stack_top_position = stack[-1].position if stack else [0,0]
        dist_cost = dist_cost(action.position,stack_top_position)
        phase_cost = phase_cost(action.action_type,'place')
        return dist_cost + phase_cost
```
2. done prompt 


## question
1. when to prompt? update prompt by past experience?
2. what task? what object? 
3. how to explore when the orientation unknown
4. shall we use LLM designed reward function.

# todo
1. image
2. other task and gripper
3. vild noise

<!-- # task tmplate
for 1 task: available_objects[] & task templete: put {pick_obj} on {place_obj}
LLM: rephrase to task templete and return extract goal [picked_ind, placed_ind]
VLM: extract positions from image

image&language command: -> LLM VLM
a list of position and goal index
 
VLM.find(obj_name,image)-> return position and confidance score -->
### 10.18
1. image input works without noise
2. add noise to position input will lead to strange RL behavior
3. shall we use depth only? Zero shot
4. reward function depend on the planning of LLM, shall it be generated by LLM?
5. tried exploration rate linear with alpha, but it is not working
Todo next week
1. long horizon tasks
2. depth only?
3. other way of dynamic exploration
# todo next week
1. check the influence of the image
2. dropout the postion input
## 10.19
1. dropout in Environment: reward? exploration? How?
- changing the environment to exploration behavior (inverse action?)
- drop out as behavior of environment or agent(data augementation)?
2. net arch
## 
RL improve from imperfect LLM and VLM, learn to compensate, with some difficult task and pure LLM and VLM as baseline
comparison
- letters in the bowl/box
- VLM+RL with hand craft VLM(center)
- VLM+RL with VLM+RL+LLM
## 
letter prompt
```
The first Image is a top down view of a block with shape of letter {}. A robot want to pick up this letter with a suction gripper.We want you to design a function to sample the pick position in 2D to make a stable pick for the robot.

[You should do]
1. Assuming the input image size is [28,28]
2. detect the letter, position and orientation of the letter
3. based on the position and orientation of the letter, draw a 2D probability map [28,28] for the robot to sample the pick position in python 

Rules:
1. You can use only python library(numpy, opencv), input of the function is the image of the picked object and the image of the placed object. Some threshold parameters are allowed and can be an optional input of the function,for example: threshold of grey scale. 
2. Do not always use center of mass as the pick position, you should consider the shape of the letter. For example get the contour of the letter and assign higher probability to the area inside the contour area.  
3. provide only 1 python function with brief explanation, you can not use undefined function in your code
```
place letter prompt

```
The first image is a top down view of a block with shape of letter{O}, which is already picked by robot. Center of the image is robot pick position. The second image is a top down view of a {bowl}.  A robot want to place this letter on a bowl with a suction gripper.We want you to design a function to sample the place position in 2D to make a stable place for the robot.
To illustrate the position, we assume place position [x,y], COM of picked object is [x1,y1], COM of placed object is [x2,y2], all those position is in image frame. After place, the COM of picked object would be [x1+x-w/2,y1+y-h/2] in the second image frame, translation is [x-w/2,y-h/2]. The shape of the picked object after place should have a good alignment with the shape of the placed object.

[You should do]
1. Assuming the input image size is [28,28]
2. detect the objects and position of the picked object and placed object
3. Each pixel of 2D probability map is the probability of the place position in this pixel. The robot gripper picked the object in the first image, and the center of the image is the robot pick position. The robot gripper place the object in the second image, you should find a place position in the second image that is easy to place the object to align with the {letter} with the {bowl}. 
4. based on the position and shape of the picked object and placed object, draw a 2D probability map [28,28] for the robot to sample the place position in python 

Rules:
1. You can use only python library(numpy, opencv), input of the function is the image of the picked object and the image of the placed object. Some threshold parameters are allowed and can be an optional input of the function,for example: threshold of grey scale. 
2. The place position is the pixel in the second image, the first image would always center at the placed position. A good place position should be easy to align the picked object with the placed object. 
3. provide only 1 python function with brief explanation, you can not use undefined function in your code
```

## prompt for high level
[task setting]
You should design a function for high level action planning of table-top manipulation. The input of this function is state of robot and objects. Output is the desired behavior of this robot.

[robot api]
robot api contains the available actions of robot, you can use them to design your function.

class robot:
    previous_action = []
    # previous_action is a list of previous actions of robot
    def pick(self,obj):
        # pick up the object, ONLY if the object is on the table and the robot has not picked up an object successfully on last step
        self.servo(obj.position)
        success = self.grasp(obj)
        previous_action.append(["pick",obj,success])
    def place(self,location):
        # place the object to the location, ONLY if the robot has picked up an object successfully on last step
        self.servo(location)
        success = self.release()# return True if the object is released successfully
        previous_action.append(["place",location,success])
    def get_object_in_gripper(self,):
        # return the object in gripper
        if self.object_in_gripper:
            return self.object_in_gripper
        else:
            return None

object api contains the information of objects, you can use them to design your function.

class object:
    position = [x,y]
    def on(self,object):
        # return True if this object is on the object
        if np.linalg.norm(self.position[:2]-object.position[:2]) < 0.05: # distance between object is less than 5cm
            if self.position[2] > object.position[2]: # height of this object is higher than the object
                return True
            else:
                return False
        else:
            return False
        
    
    def in_middle(self,object1,object2):
        # self defined function
        # return True if this object is in the middle of object1 and object2
        center = (object1.position[:2]+object2.position[:2])/2
        if np.linalg.norm(self.position[:2]-center) < 0.05: # distance between object is less than 5cm
            return True
        else:
            return False
    # you can define your own function here
    ....

[template]

def planner(robot,objects):
    # robot: robot state
    # objects: list of object state
    # return: action for robot to execute

    if condition not meet:
    # self defined condition for robot to detect unwanted situation
    # check robot.previous_action or object geometry relation
        if wrong_pick:
            # place it to somewhere else because robot can only pick 1 object at a time
            return robot.place(position)
        elif wrong_place:
            # pick it up and place it to correct position
            return robot.pick(object)
    if some condition meet:
        # self defined condition for robot to detect wanted situation
        # for example: for put 3 blocks in a line, if 2 blocks are already in a line, then put the third block in the line
        do something
    elif some other condition:
        do somthing else
    elif ...

def relation(object1,object2,..):
    # self defined function to get the relation between objects
    # for example: for put blocks in a line, you need to know the relation between blocks if one block is in line with of other 2 blocks
    return True/False

[example]
for languange command: put all blocks in a line

def find_highest_stack(objects):
    # Finds the stack of objects with the topmost object
    # and returns its top object (ignoring the object in the robot's gripper if any)
    highest_stack_top = None
    highest_height = -np.inf
    
    for obj in objects:
        if obj.position[2] > highest_height and not any(other_obj.on(obj) for other_obj in objects):
            highest_stack_top = obj
            highest_height = obj.position[2]
    
    return highest_stack_top
def find_highest_nonstacked_block(objects, robot,stack_top):
    # Finds the highest block that is not stacked and not currently in the robot's gripper
    gripped_object = robot.get_object_in_gripper()
    nonstacked_blocks = [obj for obj in objects if not stack_top.on(obj) ]

    highest_nonstacked_block = None
    highest_nonstacked_height = -np.inf

    for obj in nonstacked_blocks:
        if obj.position[2] > highest_nonstacked_height and not any(other_obj.on(obj) for other_obj in nonstacked_blocks):
            highest_nonstacked_block = obj
            highest_nonstacked_height = obj.position[2]

    return highest_nonstacked_block

def planner(robot, objects):
    stack_base = find_highest_stack(objects)
    
    # Find the block to move next (the highest one that is not part of the stack)
    block_to_move = find_highest_nonstacked_block(objects, robot,stack_base)
    
    # If the robot is already holding the correct block, place it onto the stack base
    if robot.get_object_in_gripper() == block_to_move:
        robot.place(stack_base.position)
    else:
        # If the robot isn't holding the correct block, pick the correct block
        robot.pick(block_to_move)
[command] put {obj} on {bowl} {obj} is from[letter L,letter V,letter O] {bowl} is from[blue bowl,green bowl,yellow bowl]
[command] put {objects} on different {corners} 
{obj} is from[letter L,letter V,letter O] 
corners is the corner of the table including[left up,left down,right up,right down] 

###[command] put {obj} on {bowl} {obj} is from[letter L,letter V,letter O] {bowl} is from[blue bowl,green bowl,yellow bowl]
```python

import numpy as np

def find_object_by_description(objects, description):
    # Helper function to find an object based on its description, such as 'letter L'.
    # This assumes that there is a way to determine each object's description, which is not outlined in the API.
    # The details of this implementation are not provided as they depend on how the descriptions are assigned to objects.
    for obj in objects:
        if obj.description == description:
            return obj
    return None

def planner(robot, objects, obj_desc, bowl_desc):
    # robot: robot state
    # objects: list of object state
    # obj_desc: description of the object to place, for example, 'letter L'
    # bowl_desc: description of the bowl, for example, 'blue bowl'
    # return: action for robot to execute

    # Find the specified object and bowl based on their descriptions
    obj_to_place = find_object_by_description(objects, obj_desc)
    target_bowl = find_object_by_description(objects, bowl_desc)

    if obj_to_place is None or target_bowl is None:
        # If either the object or the bowl cannot be found, return an error or a null action
        return None
    
    # Check if the object is already in the correct bowl
    if obj_to_place.on(target_bowl):
        # The object is already in place, no action needed
        return None
    else:
        # If the robot's gripper is not empty, release the current object
        if robot.get_object_in_gripper() is not None and robot.get_object_in_gripper() != obj_to_place:
            robot.place(some_safe_location)  # some_safe_location needs to be defined
            return
    
        # If the robot's gripper is empty or has the wrong object
        if robot.get_object_in_gripper() is None or robot.get_object_in_gripper() != obj_to_place:
            # Pick the specified object
                robot.pick(obj_to_place)
        # If the robot is now holding the correct object, place it in the bowl
        if robot.get_object_in_gripper() == obj_to_place:
                robot.place(target_bowl.position)
```
[command] put {objects} on different {corners} 
{obj} is from[letter L,letter V,letter O] 
corners is the corner of the table including[left up,left down,right up,right down] 
```python
```