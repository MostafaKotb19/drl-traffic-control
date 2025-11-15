from agent import TrafficAgent
from environment import TrafficEnv

def print_results(scenario_name, results):
    print(f"\n--- Results for {scenario_name} ---")
    for key, value in results.items():
        print(f"{key}: {value:.2f}")

if __name__ == "__main__":
    best_model_name = "ppo"
        
    final_agent_for_eval = TrafficAgent(
        train_env=TrafficEnv(), # Dummy env
        model_name=best_model_name
    )
    print(f"--- Will use model from {final_agent_for_eval.best_model_path}.zip for testing. ---")

    # === VISUAL TEST ===
    print("\n--- Running a visual test of the trained agent in Rush Hour ---")
    if final_agent_for_eval:
        final_agent_for_eval.test()