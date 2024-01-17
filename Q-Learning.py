# Group Number 10
# 20QM30004 (Naveen Sani)
# CWV2
# Variable Rewards in Cliff Walking [Version 2]

import numpy as np
from gym import Env,spaces, logger
from gym.spaces import Box, Discrete
import random
from typing import Optional
from gym.envs.toy_text.utils import categorical_sample
from os import path
from gym.error import DependencyNotInstalled
from io import StringIO
from contextlib import closing
import pickle as pkl

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class CustomEnv(Env):

    metadata = {
        "render_modes": ["human", "rgb_array", "ansi"],
        "render_fps": 4,
    }



    def __init__(self, render_mode: Optional[str] = None):

        self.step_count = 0  # Add this line to initialize the step counter

        self.shape = (7, 12)
        self.start_state_index = np.ravel_multi_index((6, 0), self.shape)

        self.nS = 84 
        self.nA = 4

        # Cliff Location
        self._cliff = np.zeros(self.shape, dtype=bool)
        self._cliff[6,1:-1] = True
        self._cliff[5,7:-1] = True
        self._cliff[5,2:5] = True
        self._cliff[4,2] = True
        self._cliff[4,10] = True
        self._cliff[4,4] = True
        self._cliff[4,2] = True
        self._cliff[3, 4] = True

        # Calculate transition probabilities and rewards
        self.P = {}
        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            self.P[s] = {a: [] for a in range(self.nA)}
            self.P[s][UP] = self._calculate_transition_prob(position, [-1, 0])
            self.P[s][RIGHT] = self._calculate_transition_prob(position, [0, 1])
            self.P[s][DOWN] = self._calculate_transition_prob(position, [1, 0])
            self.P[s][LEFT] = self._calculate_transition_prob(position, [0, -1])

        # Calculate initial state distribution
        # We always start in state (6, 0)
        self.initial_state_distrib = np.zeros(self.nS)
        self.initial_state_distrib[self.start_state_index] = 1.0

        self.observation_space = spaces.Discrete(self.nS)
        self.action_space = spaces.Discrete(self.nA)

        self.render_mode = render_mode

        # pygame utils
        self.cell_size = (60, 60)
        self.window_size = (
            self.shape[1] * self.cell_size[1],
            self.shape[0] * self.cell_size[0],
        )
        self.window_surface = None
        self.clock = None
        self.elf_images = None
        self.start_img = None
        self.goal_img = None
        self.cliff_img = None
        self.mountain_bg_img = None
        self.near_cliff_img = None
        self.tree_img = None

    def _limit_coordinates(self, coord: np.ndarray) -> np.ndarray:
        """Prevent the agent from falling out of the grid world."""
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord    
    
    def _calculate_transition_prob(self, current, delta):
        """Determine the outcome for an action. Transition Prob is always 1.0.

        Args:
            current: Current position on the grid as (row, col)
            delta: Change in position for transition

        Returns:
            Tuple of ``(1.0, new_state, reward, terminated)``
        """
        new_position = np.array(current) + np.array(delta)
        new_position = self._limit_coordinates(new_position).astype(int)
        new_state = np.ravel_multi_index(tuple(new_position), self.shape)
        if self._cliff[tuple(new_position)]:
            return [(1.0, self.start_state_index, -100, False)]

        terminal_state = (self.shape[0] - 1, self.shape[1] - 1)
        is_terminated = tuple(new_position) == terminal_state
        return [(1.0, new_state, -1, is_terminated)]
    

    def calculate_transition_values(self, a, nsteps):

        if a == UP:
            delta = [-1, 0]
        elif a == RIGHT:
            delta = [0, 1]
        elif a == DOWN:
            delta = [1, 0]
        elif a == LEFT:
            delta = [0, -1]   

        current = np.unravel_index(self.s, self.shape)    
    
        new_position = np.array(current) + np.array(delta)
        new_position = self._limit_coordinates(new_position).astype(int)
        new_state = np.ravel_multi_index(tuple(new_position), self.shape)
        if self._cliff[tuple(new_position)]:
            return [(1.0, self.start_state_index, -10*nsteps, False)]

        terminal_state = (self.shape[0] - 1, self.shape[1] - 1)
        is_terminated = tuple(new_position) == terminal_state
        if is_terminated:
            return [(1.0, new_state, 5000, is_terminated)]
        
        return [(1.0, new_state, -1, is_terminated)]

    
    def step(self, a):

        # Increment step counter
        self.step_count += 1

        [(p,s,r,t)] = self.calculate_transition_values(a, self.step_count)
        print('Step:{} Reward:{}'.format(self.step_count, r))

        self.s = s
        self.lastaction = a

        if self.render_mode == "human":
            self.render()
        return (int(s), r, t, False, {"prob": p})
    
    

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.s = categorical_sample(self.initial_state_distrib, self.np_random)
        self.lastaction = None
        self.step_count = 0
        

        if self.render_mode == "human":
            self.render()
        return int(self.s), {"prob": 1}
    
    def render(self):
        if self.render_mode is None:
            logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                #f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
        elif self.render_mode == "ansi":
            return self._render_text()
        else:
            return self._render_gui(self.render_mode)
        

    def _render_gui(self, mode):
        try:
            import pygame
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[toy_text]`"
            )
        if self.window_surface is None:
            pygame.init()

            if mode == "human":
                pygame.display.init()
                pygame.display.set_caption("CliffWalking")
                self.window_surface = pygame.display.set_mode(self.window_size)
            else:  # rgb_array
                self.window_surface = pygame.Surface(self.window_size)
        if self.clock is None:
            self.clock = pygame.time.Clock()
        if self.elf_images is None:
            hikers = [
                path.join(path.dirname(__file__), "img/elf_up.png"),
                path.join(path.dirname(__file__), "img/elf_right.png"),
                path.join(path.dirname(__file__), "img/elf_down.png"),
                path.join(path.dirname(__file__), "img/elf_left.png"),
            ]
            self.elf_images = [
                pygame.transform.scale(pygame.image.load(f_name), self.cell_size)
                for f_name in hikers
            ]
        if self.start_img is None:
            file_name = path.join(path.dirname(__file__), "img/stool.png")
            self.start_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.goal_img is None:
            file_name = path.join(path.dirname(__file__), "img/cookie.png")
            self.goal_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.mountain_bg_img is None:
            bg_imgs = [
                path.join(path.dirname(__file__), "img/mountain_bg1.png"),
                path.join(path.dirname(__file__), "img/mountain_bg2.png"),
            ]
            self.mountain_bg_img = [
                pygame.transform.scale(pygame.image.load(f_name), self.cell_size)
                for f_name in bg_imgs
            ]
        if self.near_cliff_img is None:
            near_cliff_imgs = [
                path.join(path.dirname(__file__), "img/mountain_near-cliff1.png"),
                path.join(path.dirname(__file__), "img/mountain_near-cliff2.png"),
            ]
            self.near_cliff_img = [
                pygame.transform.scale(pygame.image.load(f_name), self.cell_size)
                for f_name in near_cliff_imgs
            ]
        if self.cliff_img is None:
            file_name = path.join(path.dirname(__file__), "img/mountain_cliff.png")
            self.cliff_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )

        for s in range(self.nS):
            row, col = np.unravel_index(s, self.shape)
            pos = (col * self.cell_size[0], row * self.cell_size[1])
            check_board_mask = row % 2 ^ col % 2
            self.window_surface.blit(self.mountain_bg_img[check_board_mask], pos)


            if row < self.shape[0] - 1 and self._cliff[row + 1, col]:
                self.window_surface.blit(self.near_cliff_img[check_board_mask], pos)      
            if self._cliff[row, col]:
                self.window_surface.blit(self.cliff_img, pos)
            if s == self.start_state_index:
                self.window_surface.blit(self.start_img, pos)
            if s == self.nS - 1:
                self.window_surface.blit(self.goal_img, pos)
            if s == self.s:
                elf_pos = (pos[0], pos[1] - 0.1 * self.cell_size[1])
                last_action = self.lastaction if self.lastaction is not None else 2
                self.window_surface.blit(self.elf_images[last_action], elf_pos)

        if mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window_surface)), axes=(1, 0, 2)
            )   

    def _render_text(self):
        outfile = StringIO()

        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            if self.s == s:
                output = " x "
            # Print terminal state
            elif position == (3, 11):
                output = " T "
            elif self._cliff[position]:
                output = " C "
            else:
                output = " o "

            if position[1] == 0:
                output = output.lstrip()
            if position[1] == self.shape[1] - 1:
                output = output.rstrip()
                output += "\n"

            outfile.write(output)
        outfile.write("\n")

        with closing(outfile):
            return outfile.getvalue()     
        

        


env = CustomEnv(render_mode="human") 

# env = CustomEnv()


# # Initializing Q-table with zeros
# q_table = np.zeros(shape=(84, 4))

# # Define hyperparameters
# epsilon = 0.1  # Exploration rate
# alpha = 0.1    # Learning rate
# gamma = 0.9    # Discount factor
# num_episodes = 500

# # Define epsilon-greedy policy for action selection
# def epsilon_greedy_policy(state, exploration=0.0):
#     action = int(np.argmax(q_table[int(state)]))
#     if np.random.random() <= exploration:
#         action = int(np.random.randint(low=0, high=4, size=1))
#     return action

# # Training loop
# for episode in range(num_episodes):

#     # Initialize episode
#     done = False
#     total_reward = 0
#     episode_length = 0

#     # Reset the environment and get the initial state
#     state = env.reset()[0]

#     # Episode main loop
#     while not done:
#         # Select an action using epsilon-greedy policy
#         action = epsilon_greedy_policy(state, epsilon)

#         # Take an action in the environment
#         next_state, reward, done, _, _ = env.step(action)

#         # Select the optimal action for the next state
#         next_action = epsilon_greedy_policy(next_state)

#         # Q-Learning update
#         q_table[state][action] += alpha * (reward + gamma * q_table[next_state][next_action] - q_table[state][action])

#         # Transition to the next state
#         state = next_state

#         # Update episode statistics
#         total_reward += reward
#         episode_length += 1

#     # Print episode information
#     print("Episode:", episode, "Length:", episode_length, "Total Reward:", total_reward)







# Evaluator (Comment out when the agent is learning)
q_table = pkl.load(open("q_learning_q_table.pkl", "rb"))

# Define epsilon-greedy policy for evaluation
def evaluation_policy(state, exploration=0.0):
    action = int(np.argmax(q_table[int(state)]))
    if np.random.random() <= exploration:
        action = int(np.random.randint(low=0, high=4, size=1))
    return action

# Running 5 episodes for evaluation
NUM_EVAL_EPISODES = 2
for episode in range(NUM_EVAL_EPISODES):

    # Initialize episode
    done = False
    total_reward = 0
    episode_length = 0

    # Reset the environment and get the initial state
    state = env.reset()[0]

    # Episode main loop for evaluation
    while not done:
        # Select an action using the evaluation policy
        action = evaluation_policy(state)

        # Take an action in the environment
        next_state, reward, done, _, _ = env.step(action)

        # Transition to the next state
        state = next_state

        # Update episode statistics
        episode_length += 1
        total_reward += reward

    # Print evaluation episode information
    print("Evaluation Episode:", episode, "Length:", episode_length, "Reward:", total_reward)



env.close()


# pkl.dump(q_table, open("q_learning_q_table.pkl", "wb"))
# print("Training Complete. Q Table Saved")