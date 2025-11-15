# test.py

"""
A simple script to run a visual test of the best-trained agent.

This script initializes a TrafficAgent, which automatically finds the path
to the best model from the last training stage, and then calls its `test()`
method to run a live, rendered simulation.
"""

from typing import Dict
from agent import TrafficAgent
from environment import TrafficEnv

def print_results(scenario_name: str, results: Dict[str, float]):
    """
    Prints formatted evaluation results for a given scenario.

    Args:
        scenario_name: The name of the test scenario (e.g., "RL Agent - Normal Traffic").
        results: A dictionary of performance metrics.
    """
    print(f"\n--- Results for {scenario_name} ---")
    for key, value in results.items():
        print(f"{key}: {value:.2f}")

if __name__ == "__main__":
    best_model_name = "ppo"
 
    final_agent_for_eval = TrafficAgent(
        train_env=TrafficEnv(), # A dummy environment is sufficient to initialize the agent
        model_name=best_model_name
    )
    print(f"--- Will use model from {final_agent_for_eval.best_model_path}.zip for testing. ---")

    # === VISUAL TEST ===
    print("\n--- Running a visual test of the trained agent in Normal Rate ---")
    final_agent_for_eval.test()