# record_agent.py
import os
import time
import numpy as np

# Make sure these files exist in the same folder:
# - policy_iteration.py (contains GridMDP, PolicyIteration)
# - grid_maze_env.py (contains GridMazeEnv)  <-- keep the same module name you used locally
from policy_iteration import GridMDP, PolicyIteration
from grid_maze_env import GridMazeEnv

# Try to import imageio (preferred for mp4 via ffmpeg), otherwise fallback to Pillow for GIF
try:
    import imageio.v3 as imageio_v3  # newer API if available
    imageio = imageio_v3
    IMAGEIO_V3 = True
except Exception:
    try:
        import imageio
        IMAGEIO_V3 = False
    except Exception:
        imageio = None
        IMAGEIO_V3 = False

def random_positions(grid_size, seed=None):
    rng = np.random.default_rng(seed)
    positions = []
    while len(positions) < 5:
        pos = (int(rng.integers(0, grid_size)), int(rng.integers(0, grid_size)))
        if pos not in positions:
            positions.append(pos)
    return positions  # [agent, goal, bad1, bad2]

def record_episode(env, policy, max_steps=500, step_delay=0.0):
    """
    Run one episode following `policy` and collect a list of RGB frames from env.render().
    env must be created with render_mode="rgb_array".
    Returns: frames (list of HxWx3 uint8 arrays), final_reward
    """
    frames = []
    obs, info = env.reset()
    done = False
    steps = 0
    final_reward = None

    # capture initial frame
    arr = env.render()  # rgb_array
    if arr is not None:
        frames.append(arr.copy().astype(np.uint8))

    while not done and steps < max_steps:
        agent_x, agent_y = int(obs[0]), int(obs[1])
        s = agent_x + agent_y * env.grid_size
        a = int(policy[s])

        obs, reward, terminated, truncated, info = env.step(a)
        done = terminated or truncated
        final_reward = reward
        # capture frame after action
        arr = env.render()
        if arr is not None:
            frames.append(arr.copy().astype(np.uint8))
        steps += 1
        if step_delay > 0:
            time.sleep(step_delay)

    return frames, final_reward

def save_video(frames, out_path, fps=6):
    """
    Save frames (list of numpy arrays HxWx3 uint8) to mp4 using imageio (ffmpeg) if available.
    If imageio/ffmpeg not available, try saving GIF (may be larger).
    """
    if len(frames) == 0:
        raise ValueError("No frames to save.")

    # Ensure dtype uint8
    frames = [f.astype(np.uint8) for f in frames]

    # Try imageio with ffmpeg -> mp4
    if imageio is not None:
        try:
            # imageio v3 API
            if IMAGEIO_V3:
                # plugin "pyav" or "ffmpeg" will be used when writing .mp4
                imageio.imwrite(out_path, frames, fps=fps, plugin="pyav")
            else:
                # older imageio API
                writer = imageio.get_writer(out_path, fps=fps, codec="libx264")
                for f in frames:
                    writer.append_data(f)
                writer.close()
            print(f"Saved video to: {out_path}")
            return out_path
        except Exception as e:
            print(f"imageio write failed (falling back to GIF). Error: {e}")

    # Fallback GIF using imageio if available
    if imageio is not None:
        gif_path = os.path.splitext(out_path)[0] + ".gif"
        try:
            if IMAGEIO_V3:
                imageio.imwrite(gif_path, frames, duration=1.0/fps)
            else:
                imageio.mimsave(gif_path, frames, fps=fps)
            print(f"Saved GIF to: {gif_path}")
            return gif_path
        except Exception as e:
            print(f"Fallback GIF write failed: {e}")

    # Final fallback: raise error explaining missing dependencies
    raise RuntimeError(
        "Unable to save video: imageio (with ffmpeg/pyav) not available. "
        "Install imageio-ffmpeg or imageio + ffmpeg binary to enable MP4 writing:\n"
        "pip install imageio imageio-ffmpeg\n"
    )

def main():
    grid_size = 5

    # Create env once (offscreen rendering)
    env = GridMazeEnv(render_mode="rgb_array", grid_size=grid_size)

    episodes_to_record = 5
    all_videos = []
    out_dir = "recordings"
    os.makedirs(out_dir, exist_ok=True)

    try:
        for i in range(episodes_to_record):
            # Choose a new random maze for each episode (seed=None gives randomness)
            positions = random_positions(grid_size, seed=None)
            agent_pos, goal1, goal2, bad1, bad2 = positions
            print(f"\nEpisode {i+1} positions (agent, goal1, goal2, bad1, bad2): {agent_pos}, {goal1}, {goal2}, {bad1}, {bad2}")

            # 1) Build MDP and compute policy for this layout
            mdp = GridMDP(grid_size=grid_size, goal_pos=[goal1, goal2], bad_cells=[bad1, bad2])
            pi = PolicyIteration(mdp, gamma=0.99, theta=1e-6)
            policy, V = pi.run(max_iterations=1000)
            print("  Policy computed for this maze.")

            # 2) Monkey-patch env._generate_random_positions so reset will place this layout
            def make_fixed_generate_positions(a_pos, g1, g2, b1, b2):
                def fixed_generate_positions():
                    env.agent_pos = np.array(a_pos)
                    env.goal_pos = [np.array(g1), np.array(g2)]
                    env.bad_cells = [np.array(b1), np.array(b2)]
                return fixed_generate_positions

            env._generate_random_positions = make_fixed_generate_positions(agent_pos, goal1, goal2, bad1, bad2)

            # 3) Record the episode
            frames, final_reward = record_episode(env, policy, max_steps=300, step_delay=0.0)
            print(f"  Episode {i+1}: frames={len(frames)}, final_reward={final_reward}")

            # 4) Save to disk
            timestamp = int(time.time())
            out_path = os.path.join(out_dir, f"gridmaze_test_{i+1}.mp4")
            try:
                saved_path = save_video(frames, out_path, fps=6)
                all_videos.append(saved_path)
            except Exception as e:
                print(f"  Failed to save episode {i+1}: {e}")

    finally:
        env.close()

    print("\nFinished recording. Saved files:")
    for p in all_videos:
        print(" -", p)

if __name__ == "__main__":
    main()
