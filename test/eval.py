import os
import sys
# print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environments.pickplace_environment_residual import ResPickOrPlaceEnvWithoutLangReward
from tasks.letter import PutLetterOontheBowl,PutLetterontheBowlUnseen,PutLetterontheBowl,PutLetterontheBowlUnseenLetters
import numpy as np
import matplotlib.pyplot as plt
from agents.LLMRL import LLMSAC,GuideSAC
import cv2


tasks = {
    "PutLetterontheBowl":PutLetterontheBowl,
    "PutLetterontheBowlUnseen":PutLetterontheBowlUnseen
}

def eval_model(trail,step_limit = None,model_seed = None):
    if model_seed is not None:
        model_name = "tmp/llmsac_imgatten_withnoise_PutLetterontheBowlseed"+str(model_seed)+"_model/rl_model_60000_steps.zip"
        model =LLMSAC.load(model_name)
    success = []
    steps = []

    for seed in range(trail):
        np.random.seed(seed)
        obs,_ = env.reset()
        done = False
        step = 0
        while not done:
            action,_ = model.predict(obs, deterministic=True)
            print(action)
            obs, rewards, done,_, info = env.step(action)
            if step_limit is not None:
                if step >= step_limit:
                    done = True
            if not done:
                step += 1
            if done :
                steps.append(step)
                if rewards >= 0.95:
                    success.append(1)
                else:
                    success.append(0)
                print("done")
    # RL_success = success/trial
    del model
    return success,steps

def eval_baseline(trail,step_limit = None,deterministic = True):
    success = []
    steps = []
    for seed in range(trail):
        np.random.seed(seed)
        obs,_ = env.reset()
        done = False
        step = 0
        while not done:
            action = env.get_expert_demonstration(deterministic=deterministic)
            print(action)
            obs, rewards, done,_, info = env.step(action)
            step += 1
            if step_limit is not None:
                if step >= step_limit:
                    done = True
            else:
                step += 1
            if done :
                steps.append(step)
                if rewards >= 0.95:
                    success.append(1)
                else:
                    success.append(0)
                print("done")
    return success,steps

# task = "PutLetterontheBowl"
task = "PutLetterontheBowlUnseen"
trail = 50
seeds = [0,1,2,3]
result_list = []
env = ResPickOrPlaceEnvWithoutLangReward(
                                        task= tasks[task],
                                         image_obs=True,
                                         residual=True,
                                         observation_noise=5,
                                         render=True,
                                         multi_discrete=False,
                                         scale_action=True,
                                         ee="suction",
                                         scale_obs=True,
                                         neglect_steps=False,
                                      one_hot_action = True)


for model_seed in seeds:
    success,steps = eval_model(trail,step_limit=5,model_seed=model_seed)
    dict = {
        "task":task,
        "model":"RL",
        "seed":model_seed,
        "success":success,
        "steps":steps
    }    
    result_list.append(dict)
success,steps = eval_baseline(trail,step_limit=5)
dict = {
    "task":task,
    "model":"Baseline",
    "seed":None,
    "success":success,
    "steps":steps
}
result_list.append(dict)


import pickle
file_name = "test/"+task+"_eval_model.pkl"
with open(file_name,"wb") as f:
    pickle.dump(result_list,f)
print("save success")