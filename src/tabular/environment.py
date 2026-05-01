"""5x5 stochastic gridworld environment ('Mario')."""

import random


class MarioEnvironment:
    """
    A 5x5 grid (25 cells, numbered 0-24).

    The agent starts at cell 0 and must reach cell 24 (the goal).
    Three cells are lava pits that instantly end the episode.
    A 20% 'wind' chance randomly redirects the agent's chosen move.

    Rewards:
        +20 on reaching the goal
        -20 on stepping into lava
         -1 per step otherwise (encourages short paths)
    """

    def __init__(self):
        self.grid_size = 5
        self.total_states = 25
        self.action_dimensions = 4          # 0=up, 1=down, 2=left, 3=right
        self.agent_state = 0
        self.flag_state = 24
        self.lava_states = [11, 14, 17]
        self.wind = 0.20

    def reset(self):
        """Place the agent back at the starting cell and return it."""
        self.agent_state = 0
        return self.agent_state

    def step(self, direction):
        """
        Move the agent one step in the grid.

        Parameters:
            direction (int): 0=up, 1=down, 2=left, 3=right

        Returns:
            (next_state, reward, done)
        """
        # 20% chance the move is redirected randomly
        if random.uniform(0, 1) < self.wind:
            other_directions = [m for m in range(4) if m != direction]
            direction = random.choice(other_directions)

        # Convert flat index to (row, col)
        row = self.agent_state // self.grid_size
        col = self.agent_state % self.grid_size

        # Apply movement with boundary clamping
        if direction == 0:
            row = max(0, row - 1)
        elif direction == 1:
            row = min(self.grid_size - 1, row + 1)
        elif direction == 2:
            col = max(0, col - 1)
        elif direction == 3:
            col = min(self.grid_size - 1, col + 1)

        # Convert back to flat index
        self.agent_state = row * self.grid_size + col

        if self.agent_state == self.flag_state:
            return (self.agent_state, 20, True)

        if self.agent_state in self.lava_states:
            return (self.agent_state, -20, True)

        return (self.agent_state, -1, False)
