import os
import sys
# print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environments.pickplace_environment_residual import ResPickOrPlaceEnvWithoutLangReward
from tasks.letter import PutLetterOontheBowl,PutLetterontheBowlUnseen,PutLetterontheBowl,PutLetterontheBowlUnseenLetters,PutblockonBowlSameColor
import numpy as np
import matplotlib.pyplot as plt
from agents.LLMRL import LLMSAC,GuideSAC
import cv2

env = ResPickOrPlaceEnvWithoutLangReward(
                                        task= PutblockonBowlSameColor,
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
model = LLMSAC.load("tmp/llmsac_imgatten_withnoise_PutblockonBowlSameColorseed0_model/rl_model_80000_steps.zip")

def eval_model(trail,model,step_limit = 10,model_seed = None):
    success = []
    steps = []

    for seed in range(trail):
        np.random.seed(seed)
        obs,_ = env.reset()
        print(obs.keys())
        print(env.observation_space)
        done = False
        step = 0
        while not done:
            # remove the lang_goal from the observation
            obs.pop("lang_goal")
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
print(eval_model(10,model))

