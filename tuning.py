# tuning.py

import optuna
import os
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from environment import TrafficEnv

# --- TUNING CONSTANTS ---
N_TRIALS = 50  # Number of different hyperparameter sets to try
N_TIMESTEPS_PER_TRIAL = 25000  # Timesteps to train each trial model
EVAL_FREQ = 5000 # How often to evaluate the model and check for pruning
LOGS_DIR = "tuning_logs"
os.makedirs(LOGS_DIR, exist_ok=True)


class TrialEvalCallback(EvalCallback):
    """Custom callback for Optuna that combines evaluation and pruning."""
    
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
        # This function is called periodically by the PPO learn method.
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # First, execute the evaluation logic from the parent EvalCallback.
            # This will run the evaluations and update self.last_mean_reward.
            super()._on_step()
            
            # Report the result to Optuna for this "step" (an evaluation cycle).
            self.eval_idx += 1
            self.trial.report(self.last_mean_reward, self.eval_idx)
            
            # Check if the trial should be pruned.
            if self.trial.should_prune():
                self.is_pruned = True
                # Return False to stop the training.
                return False
        # Continue training if not time for evaluation or if not pruned.
        return True


def objective(trial: optuna.Trial) -> float:
    """
    The objective function for Optuna to optimize.
    """
    print(f"\n--- Starting Trial {trial.number} ---")

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

    train_env = TrafficEnv(is_training=True)
    train_env = Monitor(train_env)
    train_vec_env = VecNormalize(DummyVecEnv([lambda: train_env]))

    eval_env_raw = TrafficEnv(is_training=False)
    eval_env_raw = Monitor(eval_env_raw)
    eval_vec_env = VecNormalize(DummyVecEnv([lambda: eval_env_raw]), training=False, norm_reward=False, norm_obs=True)

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
        model.learn(total_timesteps=N_TIMESTEPS_PER_TRIAL, callback=eval_callback)
        last_reward = eval_callback.last_mean_reward
    except AssertionError as e:
        print(f"Trial {trial.number} failed with error: {e}")
        raise optuna.exceptions.TrialPruned()
    finally:
        train_vec_env.close()
        eval_vec_env.close()

    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()

    print(f"--- Trial {trial.number} Finished. Mean Reward: {last_reward:.2f} ---")
    return last_reward


if __name__ == "__main__":
    # A trial is pruned if its value is worse than the median of past trials.
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=EVAL_FREQ // 2)
    study = optuna.create_study(direction='maximize', pruner=pruner)

    try:
        study.optimize(objective, n_trials=N_TRIALS)
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")

    print("\n\n--- OPTIMIZATION FINISHED ---")
    print(f"Number of finished trials: {len(study.trials)}")
    
    if study.best_trial:
        print("\n--- Best trial ---")
        trial = study.best_trial
        print(f"  Value (Mean Reward): {trial.value:.4f}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    '{key}': {value},")
    else:
        print("No trials were completed successfully.")

    print("\nTo use these parameters, copy them into the BEST_PARAMS dictionary in main.py")