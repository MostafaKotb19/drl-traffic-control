# tuning.py

"""
Hyperparameter tuning for the PPO agent using the Optuna library.

This script defines an "objective" function that Optuna attempts to maximize.
Each "trial" in Optuna trains a PPO model with a different set of hyperparameters
and evaluates its performance. A pruner is used to stop unpromising trials early.
"""

import optuna
import os
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from environment import TrafficEnv

# --- TUNING CONSTANTS ---
N_TRIALS = 50  # Number of different hyperparameter sets to try
N_TIMESTEPS_PER_TRIAL = 25000  # Timesteps to train each trial model
EVAL_FREQ = 5000 # How often to evaluate the model and check for pruning
LOGS_DIR = "tuning_logs"
os.makedirs(LOGS_DIR, exist_ok=True)


class TrialEvalCallback(EvalCallback):
    """
    Custom callback for Optuna that combines model evaluation with trial pruning.
    
    This callback is triggered periodically during training. It evaluates the
    model's performance, reports the score to Optuna, and checks if the trial
    should be stopped early (pruned) based on its performance relative to others.
    """
    
    def __init__(self, eval_env, trial: optuna.Trial, n_eval_episodes=5, 
                 eval_freq=10000, log_path=None, best_model_save_path=None,
                 deterministic=True):
        super().__init__(eval_env, n_eval_episodes=n_eval_episodes,
                        eval_freq=eval_freq, log_path=log_path,
                        best_model_save_path=best_model_save_path,
                        deterministic=deterministic)
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        """
        This method is called periodically by the PPO `learn` method.
        """
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # First, run the evaluation logic from the parent class.
            # This will update `self.last_mean_reward`.
            super()._on_step()
            
            # Report the latest performance to Optuna.
            self.eval_idx += 1
            self.trial.report(self.last_mean_reward, self.eval_idx)
            
            # Check with Optuna's pruner if this trial should be terminated.
            if self.trial.should_prune():
                self.is_pruned = True
                return False  # Return False to stop the training loop.
                
        return True


def objective(trial: optuna.Trial) -> float:
    """
    The objective function for Optuna to optimize.

    This function defines the search space for hyperparameters, creates a PPO
    model with a sampled configuration, trains it, and returns the final
    performance score (mean reward).

    Args:
        trial: An Optuna Trial object, used to suggest hyperparameter values.

    Returns:
        The mean reward achieved by the model on the evaluation environment.
    """
    print(f"\n--- Starting Trial {trial.number} ---")

    # Define the hyperparameter search space
    model_params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        'n_steps': trial.suggest_categorical('n_steps', [512, 1024, 2048, 4096]),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
        'gamma': trial.suggest_float('gamma', 0.95, 0.9999),
        'gae_lambda': trial.suggest_float('gae_lambda', 0.9, 0.99),
        'clip_range': trial.suggest_categorical('clip_range', [0.1, 0.2, 0.3, 0.4]),
        'ent_coef': trial.suggest_float('ent_coef', 1e-8, 1e-1, log=True),
        'vf_coef': trial.suggest_float('vf_coef', 0.2, 0.8),
    }
    
    net_arch_str = trial.suggest_categorical("net_arch", ["small", "medium"])
    net_arch = {
        "small": dict(pi=[64, 64], vf=[64, 64]),
        "medium": dict(pi=[256, 256], vf=[256, 256]),
    }[net_arch_str]
    
    policy_kwargs = dict(net_arch=net_arch, activation_fn=th.nn.ReLU)

    # Create separate training and evaluation environments
    # Note: `VecNormalize` is not used here for simplicity, but could be added.
    train_env = TrafficEnv() # Headless for speed
    train_env = Monitor(train_env)
    train_vec_env = DummyVecEnv([lambda: train_env])

    eval_env_raw = TrafficEnv()
    eval_env_raw = Monitor(eval_env_raw)
    eval_vec_env = DummyVecEnv([lambda: eval_env_raw])

    model = PPO(
        'MlpPolicy',
        train_vec_env,
        policy_kwargs=policy_kwargs,
        verbose=0,
        **model_params
    )

    eval_callback = TrialEvalCallback(
        eval_env=eval_vec_env,
        trial=trial,
        n_eval_episodes=5,
        eval_freq=EVAL_FREQ,
        deterministic=True,
        log_path=os.path.join(LOGS_DIR, f"trial_{trial.number}")
    )

    last_reward = -float('inf')
    try:
        # Train the model, with the callback handling evaluation and pruning
        model.learn(total_timesteps=N_TIMESTEPS_PER_TRIAL, callback=eval_callback)
        last_reward = eval_callback.last_mean_reward
    except AssertionError as e:
        # Catch assertion errors which can happen with invalid hyperparameter combinations in SB3
        print(f"Trial {trial.number} failed with an assertion error: {e}. Pruning trial.")
        raise optuna.exceptions.TrialPruned()
    finally:
        # Ensure environments are closed to free resources
        train_vec_env.close()
        eval_vec_env.close()

    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()

    print(f"--- Trial {trial.number} Finished. Mean Reward: {last_reward:.2f} ---")
    return last_reward