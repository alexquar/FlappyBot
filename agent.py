#flappy bird docs: https://github.com/markub3327/flappy-bird-gymnasium

import flappy_bird_gymnasium
import gymnasium
from dqn import DQN
import torch as pt
device = 'cuda' if pt.cuda.is_available() else 'cpu'
class Agent:
    def run (self, is_training=True, render=False):
        #instance of flappy bird
        env = gymnasium.make("CartPole-v1", render_mode="human" if render else None)
        # env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=False)
        num_actions = env.action_space.n
        num_states = env.observation_space.shape[0]
        # Create the DQN model
        bot = DQN(input_dim=num_states, output_dim=num_actions).to(device=device)
        #reset env
        obs, _ = env.reset()
        while True:
            # Next action:
            # 0 for nothing 1 for flap 
            action = env.action_space.sample()

            # execute action
            #obs is the observation, reward is the score, terminated is if the game is over
            obs, reward, terminated, _, info = env.step(action)
            
            # Checking if the player is still alive
            if terminated:
                break

        env.close()