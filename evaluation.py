from constants import *
from environment import TrafficEnv
from agent import TrafficAgent
import numpy as np

class TimerAgent:
    """A baseline agent that switches lights on a fixed timer."""
    def __init__(self):
        self.steps_since_last_switch = 0
        self.green_duration_steps = GREEN_DURATION / 15.0

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> tuple:
        """
        Predicts an action based on the timer.
        
        Parameters:
            obs: The current observation (not used in this agent).
            deterministic: Whether to use deterministic actions (not used here).

        Returns:
            action: 0 to keep current light, 1 to switch.
            None: Placeholder for compatibility.
        """
        self.steps_since_last_switch += 1
        if self.steps_since_last_switch > self.green_duration_steps:
            self.steps_since_last_switch = 0
            return (1, None)  # Action 1: Switch
        return (0, None)      # Action 0: Keep

def evaluate_agent(agent: TimerAgent | TrafficAgent, spawn_rate: int = None, 
                   spawn_configs: dict = None, num_episodes: int = 10) -> dict:
    """
    Runs an agent in the environment for a number of episodes and collects metrics.
    
    Parameters:
        agent: The agent to evaluate (TimerAgent or TrafficAgent).
        spawn_rate: The vehicle spawn rate for the environment.
        spawn_configs: Specific spawn configurations for different directions.
        num_episodes: Number of episodes to run for evaluation.

    Returns:
        dict: A dictionary containing evaluation metrics.
    """
    # Create a fresh environment for evaluation
    env = TrafficEnv(spawn_rate=spawn_rate, spawn_configs=spawn_configs)
    
    total_wait_time_frames = 0
    total_throughput = 0
    total_collisions = 0
    all_completed_wait_times = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            # The agent.predict() method from SB3 handles single observations correctly
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

        # At the end of the episode, collect metrics from the simulation
        total_wait_time_frames += env.sim.total_wait_time_steps
        total_throughput += env.sim.cars_passed_intersection
        total_collisions += env.collision_count
        all_completed_wait_times.extend(env.sim.completed_car_wait_times)

    # Avoid division by zero if num_episodes is 0
    if num_episodes == 0:
        return {
            "Average Wait Time per Car (seconds)": 0,
            "95th Percentile Wait Time (seconds)": 0,
            "Average Throughput (cars/episode)": 0,
            "Average Collisions per Episode": 0,
        }

    # --- Calculate Final Metrics ---
    
    # Throughput and Collisions
    avg_throughput = total_throughput / num_episodes
    avg_collisions = total_collisions / num_episodes

    # Normalized wait time: The average time each car that passed had to wait.
    # This is a much fairer metric for comparing efficiency.
    if total_throughput > 0:
        avg_wait_time_per_car_seconds = (np.mean(all_completed_wait_times)) / FPS
        p95_wait_time_seconds = np.percentile(all_completed_wait_times, 95) / FPS
    else:
        avg_wait_time_per_car_seconds = 0
        p95_wait_time_seconds = 0

    env.close()
    
    return {
        "Average Wait Time per Car (seconds)": avg_wait_time_per_car_seconds,
        "95th Percentile Wait Time (seconds)": p95_wait_time_seconds,
        "Average Throughput (cars/episode)": avg_throughput,
        "Average Collisions per Episode": avg_collisions,
    }