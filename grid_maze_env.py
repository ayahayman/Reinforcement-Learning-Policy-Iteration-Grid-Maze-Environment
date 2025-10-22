import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame


class GridMazeEnv(gym.Env):
    """
    Custom Grid Maze Environment for Reinforcement Learning
    5x5 grid with:
    - Random start position (S)
    - Random goal position (G)
    - 2 random bad cells (X)
    - Stochastic movement (70% intended, 15% perpendicular each)
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self, render_mode=None, grid_size=5):
        super().__init__()
        
        self.grid_size = grid_size
        self.render_mode = render_mode
        
        # 0=Right, 1=Up, 2=Left, 3=Down
        self.action_space = spaces.Discrete(4)
        
        # Observation space
        self.observation_space = spaces.Box(
            low=0, 
            high=grid_size-1, 
            shape=(8,), 
            dtype=np.int32
        )
        
        # Action mappings
        self.action_to_direction = {
            0: np.array([1, 0]),   # Right
            1: np.array([0, -1]),  # Up
            2: np.array([-1, 0]),  # Left
            3: np.array([0, 1])    # Down
        }
        
        # Movement probabilities
        self.intended_prob = 0.70
        self.perpendicular_prob = 0.15
        
       
        self.agent_pos = None
        self.goal_pos = None
        self.bad_cells = []
        
        # PyGame rendering variables
        self.window = None
        self.clock = None
        self.cell_size = 100
        
    def _get_perpendicular_actions(self, action):
        """Get the two perpendicular actions for a given action"""
        # For action 0 (right), perpendiculars are 1 (up) and 3 (down)
        # For action 1 (up), perpendiculars are 0 (right) and 2 (left)
        perpendiculars = {
            0: [1, 3],  # Right -> Up, Down
            1: [0, 2],  # Up -> Right, Left
            2: [1, 3],  # Left -> Up, Down
            3: [0, 2]   # Down -> Right, Left
        }
        return perpendiculars[action]
    
    def _generate_random_positions(self):
        """Generate random non-overlapping positions for S, G, and 2 X's"""
        positions = []
        while len(positions) < 4:
            pos = (
                self.np_random.integers(0, self.grid_size),
                self.np_random.integers(0, self.grid_size)
            )
            if pos not in positions:
                positions.append(pos)
        
        self.agent_pos = np.array(positions[0])
        self.goal_pos = np.array(positions[1])
        self.bad_cells = [np.array(positions[2]), np.array(positions[3])]
    
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        # Generate random positions
        self._generate_random_positions()
        
        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()
        
        return observation, info
    
    def _get_obs(self):
        """Return the observation as a flat array of coordinates"""
        return np.concatenate([
            self.agent_pos,
            self.goal_pos,
            self.bad_cells[0],
            self.bad_cells[1]
        ]).astype(np.int32)
    
    def _get_info(self):
        """Return additional info (optional)"""
        return {
            "distance_to_goal": np.linalg.norm(self.agent_pos - self.goal_pos)
        }
    
    def _is_valid_position(self, pos):
        """Check if position is within grid bounds"""
        return 0 <= pos[0] < self.grid_size and 0 <= pos[1] < self.grid_size
    
    def step(self, action):
        """Execute one step in the environment"""
        # Stochastic action selection
        rand_val = self.np_random.random()
        
        if rand_val < self.intended_prob:
            # 70% - intended action
            actual_action = action
        elif rand_val < self.intended_prob + self.perpendicular_prob:
            # 15% - first perpendicular
            actual_action = self._get_perpendicular_actions(action)[0]
        else:
            # 15% - second perpendicular
            actual_action = self._get_perpendicular_actions(action)[1]
        
        # Calculate new position
        direction = self.action_to_direction[actual_action]
        new_pos = self.agent_pos + direction
        
        # Check if new position is valid (within bounds)
        if self._is_valid_position(new_pos):
            self.agent_pos = new_pos
        # If invalid, agent stays in place (hits wall)
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if episode is done
        terminated = self._is_terminal_state()
        truncated = False
        
        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()
        
        return observation, reward, terminated, truncated, info
    
    def _calculate_reward(self):
        """
        Reward function design:
        - Reach goal: +100
        - Hit bad cell: -100
        - Each step: -1 (encourages shorter paths)
        """
        # Check if reached goal
        if np.array_equal(self.agent_pos, self.goal_pos):
            return 100.0
        
        # Check if hit bad cell
        for bad_cell in self.bad_cells:
            if np.array_equal(self.agent_pos, bad_cell):
                return -100.0
        
        # Step penalty (encourages finding shortest path)
        return -1.0
    
    def _is_terminal_state(self):
        """Check if current state is terminal (goal or bad cell)"""
        # Reached goal
        if np.array_equal(self.agent_pos, self.goal_pos):
            return True
        
        # Hit bad cell
        for bad_cell in self.bad_cells:
            if np.array_equal(self.agent_pos, bad_cell):
                return True
        
        return False
    
    def render(self):
        """Render the environment"""
        if self.render_mode == "rgb_array":
            return self._render_frame()
    
    def _render_frame(self):
        """Render the grid using PyGame"""
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.grid_size * self.cell_size, self.grid_size * self.cell_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        
        canvas = pygame.Surface((self.grid_size * self.cell_size, self.grid_size * self.cell_size))
        canvas.fill((255, 255, 255))  # White background
        
        # Draw grid lines
        for x in range(self.grid_size + 1):
            pygame.draw.line(
                canvas,
                (200, 200, 200),
                (x * self.cell_size, 0),
                (x * self.cell_size, self.grid_size * self.cell_size),
                2
            )
        for y in range(self.grid_size + 1):
            pygame.draw.line(
                canvas,
                (200, 200, 200),
                (0, y * self.cell_size),
                (self.grid_size * self.cell_size, y * self.cell_size),
                2
            )
        
        # Draw goal (Green)
        goal_rect = pygame.Rect(
            self.goal_pos[0] * self.cell_size,
            self.goal_pos[1] * self.cell_size,
            self.cell_size,
            self.cell_size
        )
        pygame.draw.rect(canvas, (0, 255, 0), goal_rect)
        
        # Draw bad cells (Red)
        for bad_cell in self.bad_cells:
            bad_rect = pygame.Rect(
                bad_cell[0] * self.cell_size,
                bad_cell[1] * self.cell_size,
                self.cell_size,
                self.cell_size
            )
            pygame.draw.rect(canvas, (255, 0, 0), bad_rect)
        
        # Draw agent (Blue circle)
        agent_center = (
            int(self.agent_pos[0] * self.cell_size + self.cell_size / 2),
            int(self.agent_pos[1] * self.cell_size + self.cell_size / 2)
        )
        pygame.draw.circle(canvas, (0, 0, 255), agent_center, self.cell_size // 3)
        
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
    
    def close(self):
        """Clean up resources"""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()