# agent.py

import os
from typing import Optional, Dict, Any
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from environment import TrafficEnv
from constants import *

class TrafficAgent:
    def __init__(self, train_env: TrafficEnv, model_name: str = "ppo_traffic_stable", model_params: Optional[Dict[str, Any]] = None):
        """
        Initializes the reinforcement learning agent for traffic light control.

        Args:
            train_env: The training environment instance.
            model_name: Base name for saving trained models and logs.
            model_params: A dictionary of hyperparameters for the PPO model.
                          If None, default parameters will be used.
        """
        # The environment is wrapped for monitoring and vectorization
        train_env = Monitor(train_env)
        self.vec_env = DummyVecEnv([lambda: train_env])

        self.models_dir = "models"
        self.logs_dir = "logs"
        self.model_name = model_name
        self.model_params = model_params if model_params is not None else {}

        # Define paths for saving the last and best versions of the model
        self.last_model_path = os.path.join(self.models_dir, f"{self.model_name}_last")
        self.best_model_dir = os.path.join(self.models_dir, f"{self.model_name}_best")
        self.best_model_path = os.path.join(self.best_model_dir, "best_model")

        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.best_model_dir, exist_ok=True)

    def train(self, total_timesteps: int, load_from_path: Optional[str] = None):
        """
        Trains the PPO model or continues training from a saved model.

        Args:
            total_timesteps: The number of timesteps for this training session.
            load_from_path: Path to a .zip model file to continue training from.
                            If None, a new model is created.
        """
        print(f"--- Starting Training Stage: {self.model_name} for {total_timesteps} timesteps ---")
        if load_from_path:
            print(f"--- Loading model from: {load_from_path} ---")
        
        # Use the same environment for training and evaluation callbacks for simplicity
        eval_vec_env = self.vec_env

        # Callback to stop training if the model does not improve for a number of evaluations
        stop_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=50, min_evals=200, verbose=1)

        # Callback to evaluate the model periodically and save the best one
        eval_callback = EvalCallback(
            eval_vec_env,
            best_model_save_path=self.best_model_dir,
            log_path=self.logs_dir,
            eval_freq=5000,
            n_eval_episodes=5,
            deterministic=True,
            render=False,
            callback_after_eval=stop_callback
        )
        
        # Default PPO hyperparameters
        default_params = {
            'policy_kwargs': dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
            'learning_rate': 0.0003, 'n_steps': 2048, 'batch_size': 64,
            'n_epochs': 10, 'gamma': 0.99, 'gae_lambda': 0.95,
            'clip_range': 0.2, 'ent_coef': 0.0, 'vf_coef': 0.5,
            'max_grad_norm': 0.5,
        }
        # Override defaults with any user-provided parameters
        default_params.update(self.model_params)
        
        # Handle special string-based net_arch parameter for convenience
        if 'net_arch' in default_params and isinstance(default_params['net_arch'], str):
            net_arch_str = default_params.pop('net_arch')
            policy_kwargs = {
                "small": dict(net_arch=dict(pi=[64, 64], vf=[64, 64])),
                "medium": dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
            }[net_arch_str]
            default_params['policy_kwargs'] = policy_kwargs

        if load_from_path:
            # Load the existing model to continue training
            model = PPO.load(load_from_path, env=self.vec_env, tensorboard_log=self.logs_dir)
        else:
            # Create a new model from scratch
            model = PPO(
                'MlpPolicy', self.vec_env, verbose=1,
                tensorboard_log=self.logs_dir, **default_params
            )

        # Start the training process
        # reset_num_timesteps=False ensures that timesteps are cumulative across training sessions
        model.learn(total_timesteps=total_timesteps, callback=eval_callback, reset_num_timesteps=False)
        model.save(self.last_model_path)

        print(f"--- Training Stage Finished. Best model saved to {self.best_model_path}.zip ---")

    def test(self):
        """
        Loads the best saved model and runs a visual test in the environment.
        The environment is set to a "normal" spawn rate for this test.
        """
        best_model_zip = self.best_model_path + ".zip"

        if not os.path.exists(best_model_zip):
            print(f"Error: Best model not found at {best_model_zip}. Please train the agent first.")
            return

        print("--- Starting Visual Test with BEST Model ---")
        # Create a new environment instance for testing with rendering enabled
        test_env_raw = TrafficEnv(render_mode="human", spawn_rate=SPAWN_RATE_NORMAL)
        test_env = DummyVecEnv([lambda: test_env_raw])
        
        model = PPO.load(self.best_model_path, env=test_env)
        obs = test_env.reset()
        
        # Loop indefinitely to allow continuous viewing
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = test_env.step(action)
            
            # Gracefully exit if the user closes the Pygame window
            if info[0].get("info") == "Window closed":
                break
            if done[0]:
                # Reset the environment when an episode ends to continue the test
                obs = test_env.reset() 
        
        test_env.close()
        print("--- Visual Test Finished ---")