# main_visual.py
import time
import numpy as np
import pygame

from policy_iteration import GridMDP, PolicyIteration, policy_to_grid
from grid_maze_env import GridMazeEnv

def random_positions(grid_size, seed=None):
    rng = np.random.default_rng(seed)
    positions = []
    while len(positions) < 5:
        pos = (int(rng.integers(0, grid_size)), int(rng.integers(0, grid_size)))
        if pos not in positions:
            positions.append(pos)
    return positions  # [agent, goal, bad1, bad2]

def print_policy_grid(policy, grid_size, goal_pos, bad_cells):
    grid = policy_to_grid(policy, grid_size)
    # goal_pos may be a single goal or a list of goals
    if isinstance(goal_pos, (list, tuple)) and len(goal_pos) == 2 and isinstance(goal_pos[0], (list, tuple)):
        goals = [tuple(goal_pos[0]), tuple(goal_pos[1])]
    else:
        try:
            goals = [tuple(goal_pos)]
        except Exception:
            goals = []
    for gx, gy in goals:
        grid[gy][gx] = "G"
    for bx, by in bad_cells:
        grid[by][bx] = "X"
    print("\nPolicy grid (rows = y from 0 at top):")
    for row in grid:
        print(" ".join(row))
    print()

def run_visual(env: GridMazeEnv, policy, episodes=5, max_steps=200, step_delay=0.2):
    """
    Run `episodes` simulations visually. `step_delay` (seconds) controls speed between steps.
    Closing the pygame window stops the run early.
    """
    successes = 0
    try:
        for ep in range(episodes):
            obs, info = env.reset()
            done = False
            steps = 0
            # Small delay before starting episode
            time.sleep(0.3)
            while not done and steps < max_steps:
                # Handle quit events (so clicking X stops everything)
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT:
                        print("Window closed by user. Exiting visual run.")
                        return successes, ep  # ep is number of completed episodes
                agent_x, agent_y = int(obs[0]), int(obs[1])
                s = agent_x + agent_y * env.grid_size
                a = int(policy[s])

                obs, reward, terminated, truncated, info = env.step(a)
                done = terminated or truncated
                steps += 1
                # Let the env render loop and then wait a little so human can follow
                time.sleep(step_delay)

            if done and reward > 0:
                successes += 1
                print(f"Episode {ep+1}: reached goal in {steps} steps.")
            elif done and reward < 0:
                print(f"Episode {ep+1}: hit bad cell in {steps} steps.")
            else:
                print(f"Episode {ep+1}: timed out after {steps} steps.")
            # short pause between episodes
            time.sleep(0.4)
    finally:
        env.close()
    return successes, episodes

def main():
    grid_size = 5
    # seed = 42  # reproducible layout and RNG

    # Choose a random maze instance (deterministic due to seed)
    positions = random_positions(grid_size)
    # positions: agent, goal1, goal2, bad1, bad2
    agent_pos, goal1, goal2, bad1, bad2 = positions
    print("Positions (agent, goal1, goal2, bad1, bad2):", agent_pos, goal1, goal2, bad1, bad2)

    # Build MDP for that fixed maze and compute policy
    mdp = GridMDP(grid_size=grid_size, goal_pos=[goal1, goal2], bad_cells=[bad1, bad2])
    pi = PolicyIteration(mdp, gamma=0.99, theta=1e-6)
    policy, V = pi.run(max_iterations=1000)

    print_policy_grid(policy, grid_size, [goal1, goal2], [bad1, bad2])

    # Print value function (optional)
    print("Value function (grid):")
    for y in range(grid_size):
        row = []
        for x in range(grid_size):
            s = x + y * grid_size
            row.append(f"{V[s]:7.1f}")
        print(" ".join(row))
    print()

    # Create environment in human mode for visual run
    env = GridMazeEnv(render_mode="human", grid_size=grid_size)

    # Use a deterministic RNG for reproducible reset outcomes
    # env.np_random = np.random.default_rng(seed)

    # Monkey-patch env._generate_random_positions so resets use our chosen layout
    def fixed_generate_positions():
        env.agent_pos = np.array(agent_pos)
        env.goal_pos = [np.array(goal1), np.array(goal2)]
        env.bad_cells = [np.array(bad1), np.array(bad2)]
    env._generate_random_positions = fixed_generate_positions

    print("Starting visual simulation. Close the window to exit early.")
    successes, total = run_visual(env, policy, episodes=5, max_steps=200, step_delay=0.18)
    print(f"\nFinished visual run: {successes}/{total} successful episodes.")
    env.close()

if __name__ == "__main__":
    main()
