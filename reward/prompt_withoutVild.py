import openai
import numpy as np
from environments.utils import xyz_to_pix
from environments.environment import BOUNDS,PIXEL_SIZE
key = "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"
openai.api_key = key

def get_description(env,action,pick_success,pos,last_pos):
    # pos:position of objects
    # last_pos: last position of objects
    # check whether the action is pick or place
    if action[0] < 0.5:
        # check whether the action is pick or place
        description = "pick up {}"
        if pick_success:
            # a meaningfull pick
            for key in env.obj_name_to_id.keys():
                if np.array(pos[key])[2] > 0.15:
                    # check height of object
                    description = description.format(key)
        else:
            description = description.format("nothing")
            # a meaningless pick         
                    
    elif action[0] > 0.5:
        # check whether the action is pick or place
        pick = ""
        for key in env.obj_name_to_id.keys():
            if np.array(last_pos[key])[2] > 0.15:
                pick = key
        if pick == "":
            description = "place nothing"
        else:
            place = "the table"
            for key in env.obj_name_to_id.keys():
                if np.linalg.norm(np.array(pos[pick])[:2]-np.array(pos[key])[:2]) < 0.05:
                    if key != pick:
                        place = key
            description = "put {} on {}".format(pick,place)
    return description

def motion_descriptor(lang_goal,info):
    found_obj = info["obj_position"][-1]
    test_prompt = 'A robot arm is attempting to manipulate an environment with the goal of ' + lang_goal+ '.\n'
    test_prompt += "the environment with size (224,224). \n"
    test_prompt += "A detector named VILD is responsible for object detection within the environment. Prior to the robot arm's actions, VILD has detected the following objects and their respective positions:"
    for i in found_obj.keys():
        pos = found_obj[i]
        pix = xyz_to_pix(pos,BOUNDS,PIXEL_SIZE)
        # pix = np.array(pix)
        if pos[2]<0.2:
            test_prompt += str(i) +" at position"+ str(pix)+ '\n'
    test_prompt += "The robot arm has already performed the following steps: \n"
    for i in range(len(info["step_result"])):
        test_prompt += str(i+1) + ". " + info["step_result"][i] + "\n"

    test_prompt += "Now, given these details, please provide the next action that the robot arm should take to fulfill the language goal. If the robot arm didn't pick up an object in the previous step, it should pick up an object. If it has already picked up an object, it should be placed."
    test_prompt += "Previous steps might have some mistake and new actions should try to correct this mistake.\n"
    # test_prompt += "reply briefly"
    # print(test_prompt)
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": test_prompt}
        ],
        temperature=0,
        top_p=1,
        )
    # print(completion["choices"][0]["message"]["content"])
    return completion["choices"][0]["message"]["content"]
def translator(motion_description,info):
    found_obj = info["obj_position"][-1]
    test_prompt = motion_description+"\n"
    test_prompt += "answer without explanation:\n"
    test_prompt += "what is the the motion primitives for this action(choose from {pick/place})?\n"
    test_prompt += "based on the motion primitives:\n"
    test_prompt += "there are some objects in the environment of size (224,224). \n"
    for i in found_obj.keys():
        pos = found_obj[i]
        pix = xyz_to_pix(pos,BOUNDS,PIXEL_SIZE)
        # pix = np.array(pix)
        if pos[2]<0.2:
            test_prompt += str(i) +" at position"+ str(pix)+ '\n'
    test_prompt += "what is the correct position for this action?\n"
    test_prompt += "For example: for action 'place object A on object B', you should return: place(position of object B). answer in format without explanation: pick/place(x,y)"
    completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": test_prompt}
    ],
    temperature=0,
    top_p=1,
    )
    result = completion["choices"][0]["message"]["content"]
    primitive,position = result.split("(")
    primitive = 0 if primitive == "pick" else 1
    position = position[:-1].split(",")
    position = [int(i) for i in position]
    action_recom = [primitive]+position

       
    
    # test_prompt = 'A robot arm is attempting to manipulate an environment with size (224,224). \n'
    # test_prompt += "there are some objects in the environment. \n"
    # for i in found_obj.keys():
    #     pos = found_obj[i]
    #     pix = xyz_to_pix(pos,BOUNDS,PIXEL_SIZE)
    #     # pix = np.array(pix)
    #     if pos[2]<0.2:
    #         test_prompt += str(i) +" at position"+ str(pix)+ '\n'
    # test_prompt += motion_description+"\n"
    # test_prompt += "The correct motion primitives for this action is "+primitives+"\n"
    # test_prompt += "what is the correct position for this action?\n"
    # test_prompt += "For example: for action 'place object A on object B', you should return: place(position of object B). answer in format without explanation: pick/place(x,y)"
    # completion = openai.ChatCompletion.create(
    # model="gpt-3.5-turbo",
    # messages=[
    #     {"role": "user", "content": test_prompt}
    # ],
    # temperature=0,
    # top_p=1,
    # )
    # print("LLM prompt:", test_prompt)
    # print("LLM motion:", motion_description)
    print("LLM recommendation",primitive,position) 
    return action_recom

def interme_reward(motion_description,action):
    # translate the motion description to pick and place position
    mid = motion_description.index("\n")
    pick_pos = motion_description[0:mid]
    place_pos = motion_description[mid+1:]
    begin_pick = pick_pos.index("[")
    end_pick = pick_pos.index("]")
    begin_place = place_pos.index("[")
    end_place = place_pos.index("]")
    pick_pos = pick_pos[begin_pick+1:end_pick]
    place_pos = place_pos[begin_place+1:end_place]
    pick_pos = pick_pos.split(",")
    place_pos = place_pos.split(",")
    pick_pos = [float(pick_pos[0]),float(pick_pos[1])]
    place_pos = [float(place_pos[0]),float(place_pos[1])]
    pick_dis = np.linalg.norm(np.array(action["pick"])-np.array(pick_pos))
    place_dis = np.linalg.norm(np.array(action["place"])-np.array(place_pos))
    print("LLM advice location",pick_pos,"*",place_pos)
    reward = 0.05/(1+pick_dis) + 0.05/(1+place_dis)
    return reward
    

if __name__ == "__main__":
    # lang_goal = "pick up the yellow block and place it in the blue bowl"
    lang_goal = "put all blocks in bowls with same color"
    found_obj = [('green bowl', [63.72969436645508, 174.53433227539062], 0.34195197),
                ('yellow bowl', [58.75020980834961, 30.1623592376709], 0.32001296), 
                ('blue bowl', [175.84609985351562, 132.44737243652344], 0.30598456), 
                ('yellow block', [79.30094909667969, 119.23040008544922], 0.27325985), 
                ('green block', [129.54827880859375, 181.69998168945312], 0.27240032), 
                ('blue block', [173.79904174804688, 47.07815170288086], 0.26218402)]
    found_obj[4] = ('green block', [65,179])

    reward = interme_reward(motion_descriptor(lang_goal,found_obj),action={"pick":[100,50],"place":[50,100]})
    print(reward)