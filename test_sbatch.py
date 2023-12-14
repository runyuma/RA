import gym
from gym import envs
from stable_baselines3 import SAC
env_list = envs.registry.keys()
env_ids = [env_spec for env_spec in env_list]
print(env_ids)
env = gym.make("Pendulum-v1")
model = SAC("MlpPolicy", env, verbose=1,device="cuda")
model.learn(total_timesteps=1000)