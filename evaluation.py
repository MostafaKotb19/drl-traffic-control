# evaluation.py

from typing import Union, Dict, Optional
import numpy as np

from constants import *
from environment import TrafficEnv
from agent import TrafficAgent

class TimerAgent:
    """A baseline agent that switches traffic lights on a fixed timer."""
    def __init__(self):
        self.steps_since_last_switch = 0
        # Convert seconds to simulation steps (1 step = 15 frames)
        self.green_duration_steps = GREEN_DURATION / 15.0

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> tuple[int, None]:
        """
        Predicts an action based on a fixed timer. Mimics the SB3 predict method.

        Args:
            obs: The current observation (unused by this agent).
            deterministic: Placeholder for API compatibility (unused).

        Returns:
            A tuple containing the action (0 for Keep, 1 for Switch) and None.
        """
        self.steps_since_last_switch += 1
        if self.steps_since_last_switch > self.green_duration_steps:
            self.steps_since_last_switch = 0
            return (1, None)  # Action 1: Switch
        return (0, None)      # Action 0: Keep

def evaluate_agent(agent: Union[TimerAgent, TrafficAgent], 
                   spawn_rate: Optional[int] = None, 
                   spawn_configs: Optional[Dict[str, int]] = None, 
                   num_episodes: int = 10) -> Dict[str, float]:
    """
    Evaluates an agent's performance in the traffic environment over multiple episodes.

    Args:
        agent: The agent to evaluate (either TimerAgent or a trained SB3 model).
        spawn_rate: The uniform vehicle spawn rate for the environment.
        spawn_configs: Specific spawn configurations for different directions.
                       Overrides `spawn_rate` if provided.
        num_episodes: The number of episodes to run for the evaluation.

    Returns:
        A dictionary containing key performance metrics, averaged over all episodes.
    """
    # Create a fresh, non-rendered environment for evaluation
    env = TrafficEnv(spawn_rate=spawn_rate, spawn_configs=spawn_configs)
    
    total_wait_time_frames = 0
    total_throughput = 0
    total_collisions = 0
    all_completed_wait_times = []  # Stores wait times of cars that successfully passed
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            # The .predict() method is compatible for both TimerAgent and SB3 models
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

        # Collect metrics from the simulation at the end of the episode
        total_wait_time_frames += env.sim.total_wait_time_steps
        total_throughput += env.sim.cars_passed_intersection
        total_collisions += env.collision_count
        all_completed_wait_times.extend(env.sim.completed_car_wait_times)

    # Avoid division by zero if evaluation is interrupted
    if num_episodes == 0:
        return {
            "Average Wait Time per Car (seconds)": 0.0,
            "95th Percentile Wait Time (seconds)": 0.0,
            "Average Throughput (cars/episode)": 0.0,
            "Average Collisions per Episode": 0.0,
        }

    # --- Calculate Final Metrics ---
    
    # Average throughput and collisions per episode
    avg_throughput = total_throughput / num_episodes
    avg_collisions = total_collisions / num_episodes

    # Calculate wait time metrics based only on cars that completed their journey.
    # This provides a more accurate measure of efficiency.
    if total_throughput > 0:
        avg_wait_time_per_car_seconds = (np.mean(all_completed_wait_times)) / FPS
        p95_wait_time_seconds = np.percentile(all_completed_wait_times, 95) / FPS
    else:
        avg_wait_time_per_car_seconds = 0.0
        p95_wait_time_seconds = 0.0

    env.close()
    
    return {
        "Average Wait Time per Car (seconds)": avg_wait_time_per_car_seconds,
        "95th Percentile Wait Time (seconds)": p95_wait_time_seconds,
        "Average Throughput (cars/episode)": avg_throughput,
        "Average Collisions per Episode": avg_collisions,
    }