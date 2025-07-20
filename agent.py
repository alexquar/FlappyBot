#flappy bird docs: https://github.com/markub3327/flappy-bird-gymnasium
#demo https://www.youtube.com/watch?v=y3BSPfmMIkA&list=PL58zEckBH8fCMIVzQCRSZVPUp3ZAVagWi&index=3
import gymnasium
from dqn import DQN
import torch as pt
from experience_replay import ReplayMemory
import itertools
import yaml
import random 
import matplotlib
import argparse
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os
import numpy as np
DATE_FORMAT = "%Y-%m-%d %H:%M:%S" 
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)
device = 'cuda' if pt.cuda.is_available() else 'cpu'
device = 'cpu'  # Force to use CPU for compatibility with all systems
matplotlib.use('Agg')  # Use 'Agg' backend for non-interactive plotting
class Agent():
    def run (self, is_training=True, render=False):
        #instance of flappy bird
        if is_training:
            start_time = datetime.now()
            last_graph_update_time = start_time
            log_message = f"Training started at {start_time.strftime(DATE_FORMAT)}"
            print(log_message)
            with open(self.LOG_FILE, 'w') as log_file:
                log_file.write(f"{start_time.strftime(DATE_FORMAT)}: {log_message}\n")
        env = gymnasium.make(self.env_id, render_mode="human" if render else None, **self.env_make_params)
        # env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=False)
        num_actions = env.action_space.n
        num_states = env.observation_space.shape[0]
        rewards_per_epoc = []
        epsilon_history = []
        # Create the DQN model
        policy_dqn = DQN(input_dim=num_states, output_dim=num_actions).to(device=device)
        
        
        if is_training:
            memory = ReplayMemory(self.replay_memory_size, seed=42)
            epsilon = self.epsilon_start
            
            target_dqn = DQN(input_dim=num_states, output_dim=num_actions).to(device=device)
            #copies the weights of the policy network to the target network
            target_dqn.load_state_dict(policy_dqn.state_dict())
            
            count = 0
            
            self.optimizer = pt.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate)
            
            best_reward = -float('inf')
        else:
            policy_dqn.load_state_dict(pt.load(self.MODEL_FILE))
            policy_dqn.eval()
        for epoc in itertools.count():
            state, _ = env.reset()
            state = pt.tensor(state, dtype=pt.float32, device=device)
            terminated = False
            epoc_reward = 0
            while not terminated and epoc_reward < self.stop_on_reward:
                
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                    action = pt.tensor(action, dtype=pt.int64, device=device)
                else:
                    with pt.no_grad():
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()
                # execute action
                #new_state is the observation, reward is the score, terminated is if the game is over
                new_state, reward, terminated, _, info = env.step(action.item())
                new_state = pt.tensor(new_state, dtype=pt.float32, device=device)
                epoc_reward += reward
                reward = pt.tensor(reward, dtype=pt.float, device=device)
                if is_training:
                    memory.append((state, action, new_state, reward, terminated))
                    
                    count+=1
                
                #Move to the next state
                state = new_state
            rewards_per_epoc.append(epoc_reward)

            if is_training:
                if epoc_reward > best_reward:
                    best_reward = epoc_reward
                    log_message = f"New best reward: {best_reward} at epoc {epoc}"
                    print(log_message)
                    with open(self.LOG_FILE, 'a') as log_file:
                        log_file.write(f"{datetime.now().strftime(DATE_FORMAT)}: {log_message}\n")
                    pt.save(policy_dqn.state_dict(), self.MODEL_FILE)
                # Update graph every x seconds
                current_time = datetime.now()
                if current_time - last_graph_update_time > timedelta(seconds=10):
                        self.save_graph(rewards_per_epoc, epsilon_history)
                        last_graph_update_time = current_time
                if len(memory) > self.batch_size:
                    batch = memory.sample(self.batch_size)
                    self.optimize(batch, policy_dqn, target_dqn)
                    epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                    epsilon_history.append(epsilon)
                
                    if count > self.sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        count = 0
                
    def optimize(self, batch, policy_dqn, target_dqn):
         # Transpose the list of experiences and separate each element
        states, actions, new_states, rewards, terminations = zip(*batch)

        # Stack tensors to create batch tensors
        # tensor([[1,2,3]])
        states = pt.stack(states)

        actions = pt.stack(actions)

        new_states = pt.stack(new_states)

        rewards = pt.tensor(rewards)
        terminations = pt.tensor(terminations).float().to(device)

        with pt.no_grad():
            if self.enable_double_dqn:
                best_actions_from_policy = policy_dqn(new_states).argmax(dim=1)

                target_q = rewards + (1-terminations) * self.discount_factor * \
                                target_dqn(new_states).gather(dim=1, index=best_actions_from_policy.unsqueeze(dim=1)).squeeze()
            else:
                # Calculate target Q values (expected returns)
                target_q = rewards + (1-terminations) * self.discount_factor * target_dqn(new_states).max(dim=1)[0]
                '''
                    target_dqn(new_states)  ==> tensor([[1,2,3],[4,5,6]])
                        .max(dim=1)         ==> torch.return_types.max(values=tensor([3,6]), indices=tensor([3, 0, 0, 1]))
                            [0]             ==> tensor([3,6])
                '''

        # Calcuate Q values from current policy
        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()
        '''
            policy_dqn(states)  ==> tensor([[1,2,3],[4,5,6]])
                actions.unsqueeze(dim=1)
                .gather(1, actions.unsqueeze(dim=1))  ==>
                    .squeeze()                    ==>
        '''

        # Compute loss
        loss = self.loss_fn(current_q, target_q)

        # Optimize the model (backpropagation)
        self.optimizer.zero_grad()  # Clear gradients
        loss.backward()             # Compute gradients
        self.optimizer.step()       # Update network parameters i.e. weights and biases
            
           
    def __init__(self, config_file='hyperparams.yml', hyperparams_set = 'cartpole1'):
        with open(config_file, 'r') as file:
            all_hyperparams = yaml.safe_load(file)
            hyperparams = all_hyperparams[hyperparams_set]
        self.env_make_params = hyperparams.get('env_make_params', {})
        self.env_id = hyperparams['env_id']
        self.replay_memory_size = hyperparams['replay_memory_size']
        self.batch_size = hyperparams['batch_size']
        self.epsilon_start = hyperparams['epsilon_start']
        self.epsilon_decay = hyperparams['epsilon_decay']
        self.epsilon_min = hyperparams['epsilon_min']
        self.sync_rate = hyperparams['sync_rate']
        self.loss_fn = pt.nn.MSELoss()
        self.optimizer = None # Placeholder
        self.learning_rate = hyperparams['learning_rate']
        self.discount_factor = hyperparams['discount_factor']
        self.enable_double_dqn = False
        self.stop_on_reward     = hyperparams['stop_on_reward']         # stop training after reaching this number of rewards
        
        self.LOG_FILE = os.path.join(RUNS_DIR, f"{hyperparams_set}.log")
        self.MODEL_FILE = os.path.join(RUNS_DIR, f"{hyperparams_set}.pt")
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f"{hyperparams_set}.png")
        
    def save_graph(self, rewards_per_episode, epsilon_history):
        # Save plots
        fig = plt.figure(1)

        # Plot average rewards (Y-axis) vs episodes (X-axis)
        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x-99):(x+1)])
        plt.subplot(121) # plot on a 1 row x 2 col grid, at cell 1
        # plt.xlabel('Episodes')
        plt.ylabel('Mean Rewards')
        plt.plot(mean_rewards)

        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        plt.subplot(122) # plot on a 1 row x 2 col grid, at cell 2
        # plt.xlabel('Time Steps')
        plt.ylabel('Epsilon Decay')
        plt.plot(epsilon_history)

        plt.subplots_adjust(wspace=1.0, hspace=1.0)

        # Save plots
        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('hyperparameters', help='')
    parser.add_argument('--train', help='Training mode', action='store_true')
    args = parser.parse_args()

    dql = Agent(hyperparams_set=args.hyperparameters)

    if args.train:
        dql.run(is_training=True)
    else:
        dql.run(is_training=False, render=True)
