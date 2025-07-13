#flappy bird docs: https://github.com/markub3327/flappy-bird-gymnasium
#demo https://www.youtube.com/watch?v=y3BSPfmMIkA&list=PL58zEckBH8fCMIVzQCRSZVPUp3ZAVagWi&index=3
import flappy_bird_gymnasium
import gymnasium
from dqn import DQN
import torch as pt
from experience_replay import ReplayMemory
import itertools
import yaml
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
        
        if is_training:
            memory = ReplayMemory(self.replay_memory_size, seed=42)
            
        #reset env
        for epoc in itertools.count():
            state, _ = env.reset()
            terminated = False
            epoc_reward = 0
            while not terminated:
                # Next action:
                # 0 for nothing 1 for flap 
                action = env.action_space.sample()

                # execute action
                #new_state is the observation, reward is the score, terminated is if the game is over
                new_state, reward, terminated, _, info = env.step(action)
                epoc_reward += reward
                if is_training:
                    memory.append((state, action, new_state, reward, terminated))
                
                #Move to the next state
                state = new_state
            print(f"Episode {epoc} finished with reward: {epoc_reward}")
            
    def __init__(self, config_file='hyperparams.yml', hyperparams_set = 'cartpole1'):
        with open(config_file, 'r') as file:
            all_hyperparams = yaml.safe_load(file)
            hyperparams = all_hyperparams[hyperparams_set]
        self.replay_memory_size = hyperparams['replay_memory_size']
        self.batch_size = hyperparams['batch_size']
        self.epsilon_start = hyperparams['epsilon_start']
        self.epsilon_decay = hyperparams['epsilon_decay']
        self.epsilon_min = hyperparams['epsilon_min']
