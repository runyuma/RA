from environments.pickplace_environment_residual import ResPickOrPlaceEnvWithoutLangReward,ResSimplifyPickOrPlaceEnvWithoutLangReward
from tasks.task import PickBowl,PutBowlInBowl
from agents.LLMRL import LLMSAC,GuideSAC
from agents.PolicyNet import CustomCombinedExtractor
import numpy as np
import matplotlib.pyplot as plt


from stable_baselines3 import SAC,PPO,TD3,DDPG,A2C,ppo
from stable_baselines3.common.callbacks import CheckpointCallback


task_name = "pickbowl"
task_name = "putbowl"
task_name = "putblockbowl"
policy_name = "llmsac_imgres"
# policy_name = "llmsac_res"
# policy_name = "sac_res"
checkpoint_callback = CheckpointCallback(
  save_freq=5000,
  save_path="tmp/"+policy_name+'_'+task_name +"_models/",
  name_prefix="rl_model",
  save_replay_buffer=True,
  save_vecnormalize=True,
)

if policy_name == "sac_res":
  env = ResSimplifyPickOrPlaceEnvWithoutLangReward(render=True,multi_discrete=False,scale_action=True,scale_obs=True,one_hot_action = True)
  model= SAC(
      "MultiInputPolicy",
      env,
      learning_rate=0.0006,
      gamma=0.9,
      # epsilon_llm=0.15,
      tensorboard_log="tmp/pickplace_tb/",
  )
  model.learn(total_timesteps=50000,callback=checkpoint_callback,tb_log_name= "Resrl_model")
elif policy_name == "llmsac_res":
  if task_name == "pickbowl":
    env = ResSimplifyPickOrPlaceEnvWithoutLangReward(task= PickBowl,render=True,multi_discrete=False,scale_action=True,scale_obs=True,one_hot_action = True)
  if task_name == "putbowl":
    env = ResSimplifyPickOrPlaceEnvWithoutLangReward(task= PutBowlInBowl,render=True,multi_discrete=False,scale_action=True,scale_obs=True,one_hot_action = True)
    print(env.task.category_names)
    print(env.observation_space)
  # env = ResSimplifyPickOrPlaceEnvWithoutLangReward(render=True,multi_discrete=False,scale_action=True,scale_obs=True,one_hot_action = True)
  ep = 0.1
  model= LLMSAC(
      "MultiInputPolicy",
      env,
      learning_rate=0.0006,
      gamma=0.9,
      epsilon_llm=ep,
      tensorboard_log="tmp/pickplace_tb/",
  )
  model.learn(total_timesteps=50000,callback=checkpoint_callback,tb_log_name= "llmResrl"+str(100*ep)+task_name+"_model")
elif policy_name == "llmsac_imgres":
  env = ResPickOrPlaceEnvWithoutLangReward(image_obs=True,render=True,multi_discrete=False,scale_action=True,ee ="suction",scale_obs=True,one_hot_action = True)
  policy_kwargs = dict(
    features_extractor_class=CustomCombinedExtractor,
    features_extractor_kwargs=dict(features_dim=128),
)

  ep = 0.10
  model= LLMSAC(
      "MultiInputPolicy",
      env,
      policy_kwargs=policy_kwargs,
      learning_rate=0.0006,
      buffer_size= 50000,
      gamma=0.9,
      epsilon_llm=ep,
      # tensorboard_log="tmp/pickplace_tb/",
  )
  model.learn(total_timesteps=50000,callback=checkpoint_callback,tb_log_name= "llmResrlimg"+str(100*ep)+task_name+"_model")

