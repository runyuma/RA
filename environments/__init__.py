from gym.envs.registration import load_env_plugins as _load_env_plugins
from gym.envs.registration import make, register, registry, spec
from gymnasium.envs.registration import register
register(
    id="SimplePickAndPlace-v0",
    entry_point="environments.rl_environment:SimplifyPickOrPlaceEnvWithoutLangReward",

)