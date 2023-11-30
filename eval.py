from environments.pickplace_environment_residual import ResPickOrPlaceEnvWithoutLangReward
import numpy as np
import matplotlib.pyplot as plt
from tasks.task import PutBlockInBowl,PickBowl,PutBowlInBowl
from tasks.letter import PutLetterontheBowl,PutLetterontheBowlUnseen
from stable_baselines3 import SAC,PPO,TD3
from agents.LLMRL import LLMSAC


residual = True
env = ResPickOrPlaceEnvWithoutLangReward(
task= PutLetterontheBowlUnseen,
image_obs=True,
residual=residual,
observation_noise=5,
render=True,
multi_discrete=False,
ee="suction",
scale_action=True,
scale_obs=True,
one_hot_action = True)
model =LLMSAC.load("tmp/llmsac_imgatten_withnoise15.0fixedPutLetterontheBowl_model/rl_model_25000_steps.zip")

# print("actor features extractor:",type(model.policy.actor.features_extractor))
# print(model.policy.actor.features_extractor)
# print("critic features extractor:",type(model.policy.critic.features_extractor))
# print(model.policy.critic.features_extractor)
# from tensorboardX import SummaryWriter
# writer = SummaryWriter('./tensorboard')
# with writer:
#     writer.add_graph(model.policy.actor.features_extractor)
#     writer.add_graph(model.policy.critic.features_extractor)


episode = 10
episode_num = 0
while episode_num<episode:
    obs,_ = env.reset()
    done = False
    while not done:
        action = model.predict(obs,deterministic=True)[0]
        # action = env.get_expert_demonstration()
        print("action:",action)
        obs,reward,done,_,_ = env.step(action)
    episode_num += 1



