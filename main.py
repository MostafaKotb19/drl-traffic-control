import os
from stable_baselines3 import PPO

from agent import TrafficAgent
from evaluation import TimerAgent, evaluate_agent
from constants import *
from environment import TrafficEnv # Import TrafficEnv

# --- BEST HYPERPARAMETERS ---
BEST_PARAMS = {
    'learning_rate': 0.0003, 
    'n_steps': 2048, 
    'batch_size': 64,
    'gamma': 0.99, 
    'gae_lambda': 0.95, 
    'clip_range': 0.2,
    'ent_coef': 0.01, 
    'vf_coef': 0.7, 
    'net_arch': 'medium'
}

# --- DEFINE THE TRAINING CURRICULUM ---
TRAINING_STAGES = [
    {
        "name": "stage1_normal",
        "timesteps": 2000000,
        "spawn_configs": {d: SPAWN_RATE_NORMAL for d in ['N', 'S', 'E', 'W']}
    },
    {
        "name": "stage2_rush_hour",
        "timesteps": 2000000,
        "spawn_configs": {d: SPAWN_RATE_RUSH_HOUR for d in ['N', 'S', 'E', 'W']}
    },
    {
        "name": "stage3_unbalanced",
        "timesteps": 2000000,
        "spawn_configs": {
            'N': SPAWN_RATE_UNBALANCED_BUSY, 'S': SPAWN_RATE_UNBALANCED_BUSY, 
            'E': SPAWN_RATE_UNBALANCED_QUIET, 'W': SPAWN_RATE_UNBALANCED_QUIET
        }
    }
]

def print_results(scenario_name, results):
    print(f"\n--- Results for {scenario_name} ---")
    for key, value in results.items():
        print(f"{key}: {value:.2f}")

if __name__ == "__main__":
    # === 1. CURRICULUM TRAINING PHASE ===
    TRAIN_MODEL = False
    EVALUATE_MODEL = True

    last_trained_model_path = None
    final_agent_for_eval = None

    if TRAIN_MODEL:
        for stage in TRAINING_STAGES:
            print(f"\n\n{'='*20} STARTING CURRICULUM: {stage['name'].upper()} {'='*20}")
            
            # Create an environment for this stage
            train_env = TrafficEnv(spawn_configs=stage['spawn_configs'])
            
            # Create an agent instance for this stage
            agent = TrafficAgent(
                train_env=train_env, 
                model_name=f"ppo_{stage['name']}", 
                model_params=BEST_PARAMS
            )
            
            # Train the agent, loading the model from the previous stage
            agent.train(
                total_timesteps=stage['timesteps'], 
                load_from_path=last_trained_model_path
            )
            
            # The model to load for the *next* stage is the best one from *this* stage
            last_trained_model_path = agent.best_model_path + ".zip"
            final_agent_for_eval = agent # Keep track of the last agent for its path
    else:
        # If not training, assume the final model exists from a previous run
        last_stage = TRAINING_STAGES[-1]
        for stage in reversed(TRAINING_STAGES):
            model_path = os.path.join(
                "models", f"ppo_{stage['name']}_best", "best_model.zip"
            )
            if os.path.exists(model_path):
                last_trained_model_path = model_path
                last_stage = stage
                break
            
        final_agent_for_eval = TrafficAgent(
            train_env=TrafficEnv(), # Dummy env
            model_name=f"ppo_{last_stage['name']}" # Name of the last stage
        )
        print(f"--- Skipping Training. Will use model from {final_agent_for_eval.best_model_path}.zip for evaluation. ---")


    # === 2. EVALUATION PHASE ===
    print("\n\n--- Starting Evaluation ---")
    if not EVALUATE_MODEL:
        print("--- Skipping Evaluation ---")
    else:
        num_eval_episodes = 20
        best_model_path = final_agent_for_eval.best_model_path + ".zip"

        if not os.path.exists(best_model_path):
             print(f"Cannot run evaluation. Trained model not found at {best_model_path}")
        else:
            print(f"Loading final trained agent from {best_model_path}...")
            rl_agent = PPO.load(best_model_path)
            timer_agent = TimerAgent()

            # --- Evaluate on Normal Traffic ---
            print("\nEvaluating on Normal Traffic...")
            rl_normal_results = evaluate_agent(rl_agent, spawn_rate=SPAWN_RATE_NORMAL, num_episodes=num_eval_episodes)
            timer_normal_results = evaluate_agent(timer_agent, spawn_rate=SPAWN_RATE_NORMAL, num_episodes=num_eval_episodes)
    
            # --- Evaluate on Rush Hour Traffic ---
            print("\nEvaluating on Rush Hour Traffic...")
            rl_rush_hour_results = evaluate_agent(rl_agent, spawn_rate=SPAWN_RATE_RUSH_HOUR, num_episodes=num_eval_episodes)
            timer_rush_hour_results = evaluate_agent(timer_agent, spawn_rate=SPAWN_RATE_RUSH_HOUR, num_episodes=num_eval_episodes)
            
            # --- Evaluate on Unbalanced Traffic ---
            print("\nEvaluating on Unbalanced Traffic (NS Busy)...")
            unbalanced_config = {'N': SPAWN_RATE_UNBALANCED_BUSY, 'S': SPAWN_RATE_UNBALANCED_BUSY, 'E': SPAWN_RATE_UNBALANCED_QUIET, 'W': SPAWN_RATE_UNBALANCED_QUIET}
            rl_unbalanced_results = evaluate_agent(rl_agent, spawn_configs=unbalanced_config, num_episodes=num_eval_episodes)
            timer_unbalanced_results = evaluate_agent(timer_agent, spawn_configs=unbalanced_config, num_episodes=num_eval_episodes)
            
            # === 3. REPORTING PHASE ===
            print("\n\n--- FINAL COMPARISON REPORT ---")
            print("=====================================================")
            print_results("RL Agent - Normal Traffic", rl_normal_results)
            print_results("Timer Agent - Normal Traffic", timer_normal_results)
            print("-----------------------------------------------------")
            print_results("RL Agent - Rush Hour", rl_rush_hour_results)
            print_results("Timer Agent - Rush Hour", timer_rush_hour_results)
            print("-----------------------------------------------------")
            print_results("RL Agent - Unbalanced (NS Busy)", rl_unbalanced_results)
            print_results("Timer Agent - Unbalanced (NS Busy)", timer_unbalanced_results)
            print("=====================================================")

    # === 4. VISUAL TEST ===
    print("\n--- Running a visual test of the trained agent in Rush Hour ---")
    if final_agent_for_eval:
        final_agent_for_eval.test()