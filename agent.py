#flappy bird docs: https://github.com/markub3327/flappy-bird-gymnasium
#demo https://www.youtube.com/watch?v=y3BSPfmMIkA&list=PL58zEckBH8fCMIVzQCRSZVPUp3ZAVagWi&index=3
import gymnasium
from dqn import DQN
import torch as pt
from experience_replay import ReplayMemory
import itertools
import yaml
import random 
device = 'cuda' if pt.cuda.is_available() else 'cpu'
class Agent():
    def run (self, is_training=True, render=False):
        #instance of flappy bird
        env = gymnasium.make("CartPole-v1", render_mode="human" if render else None)
        # env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=False)
        num_actions = env.action_space.n
        num_states = env.observation_space.shape[0]
        rewards_per_epoc = []
        epsilon_history = []
        # Create the DQN model
        bot = DQN(input_dim=num_states, output_dim=num_actions).to(device=device)
        
        if is_training:
            memory = ReplayMemory(self.replay_memory_size, seed=42)
            epsilon = self.epsilon_start
            
        #reset env
        for epoc in itertools.count():
            state, _ = env.reset()
            state = pt.tensor(state, dtype=pt.float32, device=device)
            terminated = False
            epoc_reward = 0
            while not terminated:
                
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                    action = pt.tensor(action, dtype=pt.int64, device=device)
                else:
                    with pt.no_grad():
                        action = bot(state.unsqueeze(dim=0)).squeeze().argmax()
                # execute action
                #new_state is the observation, reward is the score, terminated is if the game is over
                new_state, reward, terminated, _, info = env.step(action.item())
                new_state = pt.tensor(new_state, dtype=pt.float32, device=device)
                epoc_reward += reward
                if is_training:
                    memory.append((state, action, new_state, reward, terminated))
                
                #Move to the next state
                state = new_state
            print(f"Episode {epoc} finished with reward: {epoc_reward}")
            rewards_per_epoc.append(epoc_reward)
            epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
            epsilon_history.append(epsilon)
            
    def __init__(self, config_file='hyperparams.yml', hyperparams_set = 'cartpole1'):
        with open(config_file, 'r') as file:
            all_hyperparams = yaml.safe_load(file)
            hyperparams = all_hyperparams[hyperparams_set]
        self.replay_memory_size = hyperparams['replay_memory_size']
        self.batch_size = hyperparams['batch_size']
        self.epsilon_start = hyperparams['epsilon_start']
        self.epsilon_decay = hyperparams['epsilon_decay']
        self.epsilon_min = hyperparams['epsilon_min']

if __name__ == "__main__":
    agent = Agent(hyperparams_set='cartpole1')
    agent.run(is_training=True, render=True)
