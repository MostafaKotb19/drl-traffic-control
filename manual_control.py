import pygame
from environment import TrafficEnv
from constants import *

def run_manual_control():
    """
    Runs the traffic simulation in manual control mode.
    The user can switch the traffic lights by pressing the RETURN (ENTER) key.
    The simulation is configured for a "rush hour" traffic scenario.

    This script takes direct control of the simulation and rendering loop
    to ensure user input is handled correctly and immediately.
    """
    print("--- Manual Traffic Control ---")
    print("Starting simulation in rush hour mode.")
    print("Press RETURN (ENTER) to switch the traffic lights.")
    print("Close the window to exit the program.")

    # Initialize the environment with rendering enabled and set to rush hour traffic.
    # The environment object will manage the simulation state and GUI.
    env = TrafficEnv(render_mode="human", spawn_rate=SPAWN_RATE_NORMAL)
    
    # Reset the environment to set up the initial simulation state.
    env.reset()

    running = True
    # This is now a per-frame loop, not a per-agent-step loop.
    while running:
        # --- 1. Handle User Input ---
        # This is the ONLY place we check for events.
        for event in pygame.event.get():
            # Handle window close event
            if event.type == pygame.QUIT:
                running = False
            
            # Handle keyboard press event
            if event.type == pygame.KEYDOWN:
                # Check if the pressed key is RETURN (ENTER)
                if event.key == pygame.K_RETURN:
                    # Action 1: Switch the lights
                    # We directly call the simulation's control method.
                    env.sim.control_traffic_lights(1) 
                    print("RETURN pressed: Switching lights.")

        # --- 2. Update Simulation State ---
        # We call the simulation's update method directly to advance it by one frame.
        env.sim.update()
        
        # --- 3. Check for Collisions (Optional but good practice) ---
        if env.sim.check_collisions():
            print("Collision detected! Resetting simulation.")
            env.reset()

        # --- 4. Render the Frame ---
        # We use the environment's GUI object to draw the new state.
        if env.gui:
            env.gui.draw()
            # Control the frame rate to match the simulation's intended speed.
            env.gui.clock.tick(FPS)
        else:
            # If the GUI was somehow closed, exit the loop.
            running = False


    # Cleanly close the environment and Pygame window
    env.close()
    print("--- Simulation closed. ---")


if __name__ == "__main__":
    run_manual_control()