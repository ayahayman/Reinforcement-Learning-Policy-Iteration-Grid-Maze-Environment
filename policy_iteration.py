# policy_iteration.py
import numpy as np

# Action encoding: 0=Right,1=Up,2=Left,3=Down
ACTION_TO_DIR = {
    0: np.array([1, 0]),   # Right
    1: np.array([0, -1]),  # Up
    2: np.array([-1, 0]),  # Left
    3: np.array([0, 1])    # Down
}

INTENDED_PROB = 0.70
PERP_PROB = 0.15  # for each perpendicular action

def get_perpendicular_actions(a):
    return {
        0: [1, 3],
        1: [0, 2],
        2: [1, 3],
        3: [0, 2]
    }[a]

class GridMDP:
    def __init__(self, grid_size=5, goal_pos=(4,4), bad_cells=None):
        """
        Build an MDP model for the agent's position only (agent moves on grid_size x grid_size).
        Terminal states = goal_pos and bad_cells (absorbing).
        """
        self.grid_size = grid_size
        self.n_states = grid_size * grid_size
        self.n_actions = 4

        self.goal_pos = tuple(goal_pos)
        self.bad_cells = [tuple(b) for b in (bad_cells or [])]

        # mapping from (x,y) -> state index
        self.pos_to_state = {}
        self.state_to_pos = {}
        idx = 0
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                self.pos_to_state[(x, y)] = idx
                self.state_to_pos[idx] = (x, y)
                idx += 1

        # Precompute transition probabilities P[s, a] = list of (prob, s', reward)
        self.P = {s: {a: [] for a in range(self.n_actions)} for s in range(self.n_states)}
        self._build_transitions()

    def _is_valid(self, pos):
        x, y = pos
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size

    def _is_terminal_pos(self, pos):
        return tuple(pos) == self.goal_pos or tuple(pos) in self.bad_cells

    def _reward_for_pos(self, pos):
        if tuple(pos) == self.goal_pos:
            return 100.0
        if tuple(pos) in self.bad_cells:
            return -100.0
        return -1.0

    def _build_transitions(self):
        for s in range(self.n_states):
            pos = self.state_to_pos[s]
            # If s is terminal, it is absorbing: action leads to itself with reward according to that terminal
            if self._is_terminal_pos(pos):
                for a in range(self.n_actions):
                    r = self._reward_for_pos(pos)
                    self.P[s][a] = [(1.0, s, r)]
                continue

            # Non-terminal: build stochastic outcomes
            for a in range(self.n_actions):
                outcomes = {}  # s' -> accumulated prob
                # intended
                intended_dir = ACTION_TO_DIR[a]
                for prob, dir_action in [(INTENDED_PROB, intended_dir),
                                         (PERP_PROB, ACTION_TO_DIR[get_perpendicular_actions(a)[0]]),
                                         (PERP_PROB, ACTION_TO_DIR[get_perpendicular_actions(a)[1]])]:
                    new_pos = (pos[0] + int(dir_action[0]), pos[1] + int(dir_action[1]))
                    if not self._is_valid(new_pos):
                        # hits wall -> stays in place
                        s_prime = s
                        r = self._reward_for_pos(pos)  # reward of staying (no move)
                    else:
                        s_prime = self.pos_to_state[new_pos]
                        r = self._reward_for_pos(new_pos)

                    outcomes.setdefault(s_prime, 0.0)
                    outcomes[s_prime] += prob

                # convert to list of (prob, s', reward)
                self.P[s][a] = [(p, s_prime, self._reward_for_pos(self.state_to_pos[s_prime])) for s_prime, p in outcomes.items()]

class PolicyIteration:
    def __init__(self, mdp: GridMDP, gamma=0.99, theta=1e-6, eval_iters=1000):
        self.mdp = mdp
        self.gamma = gamma
        self.theta = theta
        self.eval_iters = eval_iters

        self.V = np.zeros(self.mdp.n_states, dtype=float)
        # start with uniform random policy
        self.policy = np.zeros(self.mdp.n_states, dtype=int)

    def policy_evaluation(self):
        """
        Iterative policy evaluation: Bellman expectation for fixed policy.
        Updates self.V in-place until convergence.
        """
        V = self.V
        P = self.mdp.P
        policy = self.policy
        gamma = self.gamma
        theta = self.theta

        while True:
            delta = 0.0
            for s in range(self.mdp.n_states):
                a = policy[s]
                v_new = 0.0
                for (prob, s_prime, r) in P[s][a]:
                    v_new += prob * (r + gamma * V[s_prime])
                delta = max(delta, abs(v_new - V[s]))
                V[s] = v_new
            if delta < theta:
                break
        self.V = V

    def policy_improvement(self):
        """
        Given current value function self.V, improve policy greedily.
        Returns True if policy is stable (no change), False otherwise.
        """
        policy_stable = True
        P = self.mdp.P
        V = self.V
        gamma = self.gamma

        for s in range(self.mdp.n_states):
            old_action = self.policy[s]

            # compute action-values for all actions
            action_values = np.zeros(self.mdp.n_actions, dtype=float)
            for a in range(self.mdp.n_actions):
                q = 0.0
                for (prob, s_prime, r) in P[s][a]:
                    q += prob * (r + gamma * V[s_prime])
                action_values[a] = q

            best_action = int(np.argmax(action_values))
            self.policy[s] = best_action
            if best_action != old_action:
                policy_stable = False

        return policy_stable

    def run(self, max_iterations=1000):
        """
        Main policy iteration loop.
        Returns (policy, V)
        """
        for i in range(max_iterations):
            self.policy_evaluation()
            stable = self.policy_improvement()
            if stable:
                break
        return self.policy.copy(), self.V.copy()

# Utility to visualize policy as arrows
def policy_to_grid(policy, grid_size):
    arrow = {0: "→", 1: "↑", 2: "←", 3: "↓"}
    grid = [[" " for _ in range(grid_size)] for _ in range(grid_size)]
    for s, a in enumerate(policy):
        x, y = s % grid_size, s // grid_size
        grid[y][x] = arrow[a]
    return grid
