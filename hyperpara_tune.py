from stable_baselines3 import SAC
from environments.rl_environment import TestEnv,PickOrPlaceEnvWithoutLangReward,SimplifyPickOrPlaceEnvWithoutLangReward
from tasks.task import PutBlockInBowl
import gym
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.visualization import plot_optimization_history, plot_param_importances

from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from typing import Any, Dict
import torch as th
import torch.nn as nn
# env = gym.make("SimplePickAndPlace-v0")
# ENV_ID = gym.make("SimplePickAndPlace-v0")
ENV_ID = "SimplePickAndPlace-v0"
# ENV_ID = SimplifyPickOrPlaceEnvWithoutLangReward()
# ENV_ID = "Pendulum-v1"

DEFAULT_HYPERPARAMS = {
    "policy": "MultiInputPolicy",
    # "policy": "MlpPolicy",
    "env": ENV_ID,
    "learning_starts":0,
    "batch_size":128,
    "gradient_steps":-1,
    "device":"cuda",
}
N_TIMESTEPS = 1000/10
N_STARTUP_TRIALS = 3
N_EVALUATIONS = 2
N_TRIALS = 100  # Maximum number of trials
N_JOBS = 1 # Number of jobs to run in parallel
TIMEOUT = int(60 * 120)  # 15 minutes

def sample_sac_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for A2C hyperparameters.

    :param trial: Optuna trial object
    :return: The sampled hyperparameters for the given trial.
    """
    ### YOUR CODE HERE
    # TODO:
    tau = trial.suggest_float("tau", 0.005, 0.5, log=True)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
    net_arch = trial.suggest_categorical("net_arch", ["small", "medium", "large"])
    if net_arch == "small":
        net_arch  = [64,64]
    elif net_arch == "medium":
        net_arch  = [64,64,64]
    elif net_arch == "large":
        net_arch  = [64,64,64,64]

    


    return {
        "learning_starts": 0,
        "learning_rate": learning_rate,
        'tau': tau,
        "policy_kwargs": {
            "net_arch": net_arch,
            "activation_fn": nn.ReLU
        },
    }
class TrialEvalCallback(EvalCallback):
    """
    Callback used for evaluating and reporting a trial.

    :param eval_env: Evaluation environement
    :param trial: Optuna trial object
    :param n_eval_episodes: Number of evaluation episodes
    :param eval_freq:   Evaluate the agent every ``eval_freq`` call of the callback.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic policy.
    :param verbose:
    """

    def __init__(
        self,
        eval_env: gym.Env,
        trial: optuna.Trial,
        n_eval_episodes: int = 2,
        eval_freq: int = 10,
        deterministic: bool = True,
        verbose: int = 0,
    ):

        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Evaluate policy (done in the parent class)
            super()._on_step()
            self.eval_idx += 1
            # Send report to Optuna
            self.trial.report(self.last_mean_reward, self.eval_idx)
            # Prune trial if need
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True
def objective(trial: optuna.Trial) -> float:
    """
    Objective function using by Optuna to evaluate
    one configuration (i.e., one set of hyperparameters).
    Given a trial object, it will sample hyperparameters,
    evaluate it and report the result (mean episodic reward after training)
    :param trial: Optuna trial object
    :return: Mean episodic reward after training
    """
    kwargs = DEFAULT_HYPERPARAMS.copy()

    # 1. Sample hyperparameters and update the keyword arguments
    kwargs.update(sample_sac_params(trial))
    # Create the RL model
    model = SAC(**kwargs)

    # 2. Create envs used for evaluation using `make_vec_env`, `ENV_ID` and `N_EVAL_ENVS`
    eval_env = make_vec_env(ENV_ID,1)
    # 3. Create the `TrialEvalCallback` callback defined above that will periodically evaluate
    # and report the performance using `N_EVAL_EPISODES` every `EVAL_FREQ`
    # TrialEvalCallback signature:
    # TrialEvalCallback(eval_env, trial, n_eval_episodes, eval_freq, deterministic, verbose)
    eval_callback = TrialEvalCallback(eval_env,trial)

    ### END OF YOUR CODE

    nan_encountered = False
    try:
        # Train the model
        model.learn(N_TIMESTEPS, callback=eval_callback)
    except AssertionError as e:
        # Sometimes, random hyperparams can generate NaN
        print(e)
        nan_encountered = True
    finally:
        # Free memory
        model.env.close()
        eval_env.close()

    # Tell the optimizer that the trial failed
    if nan_encountered:
        return float("nan")

    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()

    return eval_callback.last_mean_reward

th.set_num_threads(1)
# Select the sampler, can be random, TPESampler, CMAES, ...
sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
# Do not prune before 1/3 of the max budget is used
pruner = MedianPruner(
    n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=N_EVALUATIONS // 3
)
# Create the study and start the hyperparameter optimization
study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize")

try:
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=N_JOBS, timeout=TIMEOUT)
except KeyboardInterrupt:
    pass

print("Number of finished trials: ", len(study.trials))

print("Best trial:")
trial = study.best_trial

print(f"  Value: {trial.value}")

print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

print("  User attrs:")
for key, value in trial.user_attrs.items():
    print(f"    {key}: {value}")

# Write report
study.trials_dataframe().to_csv("study_results_a2c_cartpole.csv")

fig1 = plot_optimization_history(study)
fig2 = plot_param_importances(study)

fig1.show()
fig2.show()