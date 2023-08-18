import openai
import numpy as np
key = "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"
openai.api_key = key

def get_prompt(lang_goal,obj_bp,obj_ap,bounding_box= True):
    test_prompt = 'A robot arm is trying to manipulate the environment with the language goal: ' + lang_goal+ '.\n'
    test_prompt += 'A detector named VILD is trying to detect the objects in the environment(size [224,224]) from top-down view.. \n'
    test_prompt += 'before the robot arm picks up the object, VILD detects: \n' 
    for i in range(len(obj_bp)):
        if bounding_box:
            test_prompt += str(obj_bp[i][0]) +" at position"+ str(obj_bp[i][1])+ " with bounding box: "+str(obj_bp[i][2]) +'\n'
        else:
            test_prompt += str(obj_bp[i][0]) +" at position"+ str(obj_bp[i][1]) +'\n'
    test_prompt += 'after the robot arm place the object, VILD detects: \n'
    for i in range(len(obj_ap)):
        if bounding_box:
            test_prompt += str(obj_ap[i][0]) +" at position"+ str(obj_ap[i][1]) + " with bounding box: "+str(obj_ap[i][2])+'\n'
        else:
            test_prompt += str(obj_ap[i][0]) +" at position"+ str(obj_ap[i][1]) +'\n'
    test_prompt += 'do you think the robot arm successfully finish the languange goal? answer only yes or no.\n'
    test_prompt += "the relation {on} in languange goal means 2 objects have lot of overlap in top-down view. you should consider the overlap base on bounding box\n"
    #####################
    # test_prompt = 'A robot arm is trying to manipulate the environment with the language goal: ' + lang_goal+ '.\n'
    # test_prompt += 'A detector named VILD is trying to detect the objects in the environment(size [224,224]) from top-up view. \n'
    # test_prompt += 'We should deduce the behavior of robot based on those detection result. \n'
    # test_prompt += 'before the robot arm picks up the object, VILD detects: \n' 
    # for i in range(len(obj_bp)):
    #     test_prompt += str(obj_bp[i][0]) +" at postion "+ str(obj_bp[i][1]) + " with size "+str(obj_bp[i][2])+'\n'
    # test_prompt += 'after the robot arm place the object, VILD detects: \n'
    # for i in range(len(obj_ap)):
    #     test_prompt += str(obj_ap[i][0]) +" at postion "+ str(obj_ap[i][1]) + " with  size "+str(obj_ap[i][2])+ '\n'
    # test_prompt += 'do you think the robot arm successfully finish the languange goal given the observation? answer only yes or no.\n'
    # test_prompt += "Here are some rules for you to follow: \n"
    # test_prompt += "1.from the given observation, you should reason which objects have been picked and where it s placied (a region).\n"
    # test_prompt += "2.you should consider where is the right region to place given the languange goal. \n"
    # test_prompt += "3.you should consider the logic of the objects. \n"
    # test_prompt += "4.robot arm do not need to interact with the placed objects. \n"
    # test_prompt += "5. the region of the object is a rectangle with given width and height centered at position [x,y]. you should consider whether a objects is in the region. \n"
    # test_prompt += "6. answer only yes or no. \n"
    return test_prompt

def ask_LLM(text):
    
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": text}
        ],
            temperature=2,
        )
    if completion["choices"][0]["message"]["content"] == "Yes.":
        success_pred = True
    else:
        success_pred = False
    print("success_pred:",success_pred,completion["choices"][0]["message"]["content"])
    return success_pred

def motion_descriptor(lang_goal,found_obj):
    test_prompt = 'A robot arm is trying to manipulate the environment with the language goal: ' + lang_goal+ '.\n'
    test_prompt += "the environment with size (224,224) \n"
    test_prompt += 'A detector named VILD is trying to detect the objects in the environment. \n'
    test_prompt += 'before the robot arm picks up the object, VILD detects: \n' 
    for i in range(len(found_obj)):
        test_prompt += str(found_obj[i][0]) +" at position"+ str(found_obj[i][1]) + '\n'
    test_prompt += " I have questions for you: \n"
    test_prompt += "1. what is the position the robot should pick? \n"
    test_prompt += "2. what is the position the robot should place? \n"
    # test_prompt += "3. what is the motion primitive the robot should use? (choose from{'pick&place','push','drag'}) \n"
    test_prompt += "reply just the postion without other explanation \n"
    test_prompt += "reply in the format of: 1.pick position: {pick location} 2. place position: {place location} \n"
    print(test_prompt)
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": test_prompt}
        ],
        temperature=0
        
        )
    return completion["choices"][0]["message"]["content"]
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