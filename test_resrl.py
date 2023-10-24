from environments.pickplace_environment_residual import ResPickOrPlaceEnvWithoutLangReward,ResSimplifyPickOrPlaceEnvWithoutLangReward
from tasks.task import PickBowl,PutBowlInBowl
from agents.LLMRL import LLMSAC,GuideSAC
from agents.PolicyNet import CustomCombinedExtractor,FeatureExtractor
import numpy as np
import matplotlib.pyplot as plt


from stable_baselines3 import SAC,PPO,TD3,DDPG,A2C,ppo
from stable_baselines3.common.callbacks import CheckpointCallback


task_name = "pickbowl"
task_name = "putbowl"
task_name = "putblockbowl"
policy_name = "llmsac_imgdrop"
# policy_name = "llmsac_imgres"
# policy_name = "llmsac_res"
# policy_name = "sac_res"


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
  checkpoint_callback = CheckpointCallback(
  save_freq=5000,
  save_path="tmp/"+policy_name+'_'+task_name +"_models/",
  name_prefix="rl_model",
  save_replay_buffer=True,
  save_vecnormalize=True,
)
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
  
  policy_name = "llmsac_imgres_withnoise"
  ep = (0.10,"fixed")
  ep = (0.4, "auto")
  name = policy_name + str(100*ep[0])+ep[1]+task_name+"_model"
  checkpoint_callback = CheckpointCallback(
  save_freq=5000,
  save_path="tmp/"+ name +"/",
  name_prefix="rl_model",
  save_replay_buffer=False,
  save_vecnormalize=True,
)
  env = ResPickOrPlaceEnvWithoutLangReward(image_obs=True,observation_noise=5,render=True,multi_discrete=False,scale_action=True,ee ="suction",scale_obs=True,one_hot_action = True)
  policy_kwargs = dict(
    features_extractor_class=CustomCombinedExtractor,
    features_extractor_kwargs=dict(features_dim=128),
)


  model= LLMSAC(
      "MultiInputPolicy",
      env,
      policy_kwargs=policy_kwargs,
      learning_starts=0,
      learning_rate=3e-4,
      buffer_size= 80000,
      gamma=0.9,
      epsilon_llm=ep,
      tensorboard_log="tmp/pickplace_tb/",
  )

  model.learn(total_timesteps=80000,callback=checkpoint_callback,tb_log_name= name)
elif policy_name == "llmsac_imgdrop":
  
  policy_name = "llmsac_imgdrop_withnoise"
  ep = (0.15,"fixed")
  name = policy_name + str(100*ep[0])+ep[1]+task_name+"_model"
  checkpoint_callback = CheckpointCallback(
  save_freq=5000,
  save_path="tmp/"+ name +"/",
  name_prefix="rl_model",
  save_replay_buffer=False,
  save_vecnormalize=True,
)
  residual = True
  env = ResPickOrPlaceEnvWithoutLangReward(
    image_obs=True,
    residual=residual,
    observation_noise=5,
    render=True,
    multi_discrete=False,
    scale_action=True,
    ee ="suction",
    scale_obs=True,
    one_hot_action = True)
  policy_kwargs = dict(
    features_extractor_class=FeatureExtractor,
    features_extractor_kwargs=dict(features_dim=32),
  )
  print(env.observation_space)
  print(env.action_space)

  model= LLMSAC(
      "MultiInputPolicy",
      env,
      policy_kwargs=policy_kwargs,
      # learning_starts=0,
      learning_rate=3e-4,
      buffer_size= 80000,
      gamma=0.9,
      epsilon_llm=ep,
      dropout_position=0.2,
      residual_rl= residual,
      tensorboard_log="tmp/pickplace_tb/",
  )

  model.learn(total_timesteps=80000,callback=checkpoint_callback,tb_log_name= name)


