# agent.py
# Deep Q-Network agent to train and test on Flappy Bird or any Gymnasium environment.
# Inspired by: https://github.com/markub3327/flappy-bird-gymnasium
# Demo: https://www.youtube.com/watch?v=y3BSPfmMIkA

import gymnasium  # Main interface to the reinforcement learning environment
from dqn import DQN  # Our deep Q-network implementation
import torch as pt  # PyTorch for building and training neural networks
from experience_replay import ReplayMemory  # Experience Replay Buffer
import itertools  # For infinite loops over episodes
import yaml  # Load hyperparameters from config file
import random  # For epsilon-greedy exploration
import matplotlib  # For plotting training graphs
import argparse  # For command-line arguments
from datetime import datetime, timedelta  # For timestamps and timing logic
import matplotlib.pyplot as plt  # For plotting reward/epsilon graphs
import os  # For file operations
import flappy_bird_gymnasium  # Import the Flappy Bird environment
import numpy as np  # For numerical operations

# Date format used for log timestamps
DATE_FORMAT = "%Y-%m-%d %H:%M:%S" 
RUNS_DIR = "runs"  # Directory to store models, logs, and graphs
os.makedirs(RUNS_DIR, exist_ok=True)

# Set device for model execution (GPU if available, else CPU)
device = 'cuda' if pt.cuda.is_available() else 'cpu'
device = 'cpu'  # Force CPU to maximize compatibility

# Use non-interactive backend for matplotlib (required on headless systems)
matplotlib.use('Agg')

class Agent():
    def __init__(self, config_file='hyperparams.yml', hyperparams_set='cartpole1'):
        # Load hyperparameters from YAML configuration file
        with open(config_file, 'r') as file:
            all_hyperparams = yaml.safe_load(file)
            hyperparams = all_hyperparams[hyperparams_set]

        # Extract and store parameters
        self.env_make_params = hyperparams.get('env_make_params', {})
        self.env_id = hyperparams['env_id']
        self.replay_memory_size = hyperparams['replay_memory_size']
        self.batch_size = hyperparams['batch_size']
        self.epsilon_start = hyperparams['epsilon_start']
        self.epsilon_decay = hyperparams['epsilon_decay']
        self.epsilon_min = hyperparams['epsilon_min']
        self.sync_rate = hyperparams['sync_rate']  # Frequency for updating target network
        self.loss_fn = pt.nn.MSELoss()  # Mean Squared Error for Q-value regression
        self.optimizer = None  # Will be initialized later
        self.learning_rate = hyperparams['learning_rate']
        self.discount_factor = hyperparams['discount_factor']  # Gamma in Bellman equation
        self.enable_double_dqn = hyperparams['enable_double_dqn']  # Use Double DQN if True
        self.stop_on_reward = hyperparams['stop_on_reward']  # Stop training when this reward is reached
        self.fc1_nodes = hyperparams['fc1_nodes']  # Number of nodes in first hidden layer
        self.enable_dueling_dqn = hyperparams.get('enable_dueling_dqn', True)  # Use Dueling DQN if True

        # Paths for logging, saving model, and plotting
        self.LOG_FILE = os.path.join(RUNS_DIR, f"{hyperparams_set}.log")
        self.MODEL_FILE = os.path.join(RUNS_DIR, f"{hyperparams_set}.pt")
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f"{hyperparams_set}.png")

    def run(self, is_training=True, render=False):
        """
        Main loop to train or evaluate the DQN agent.
        If `is_training` is True, the model trains and saves progress.
        If `is_training` is False, it loads a trained model and runs without learning.
        """
        # If training, start timer and logging
        if is_training:
            start_time = datetime.now()
            last_graph_update_time = start_time
            log_message = f"Training started at {start_time.strftime(DATE_FORMAT)}"
            print(log_message)
            with open(self.LOG_FILE, 'w') as log_file:
                log_file.write(f"{start_time.strftime(DATE_FORMAT)}: {log_message}\n")

        # Create environment
        env = gymnasium.make(self.env_id, render_mode="human" if render else None, **self.env_make_params)
        num_actions = env.action_space.n
        num_states = env.observation_space.shape[0]

        # Reward and exploration tracking
        rewards_per_epoc = []
        epsilon_history = []

        # Initialize policy DQN
        policy_dqn = DQN(num_states, num_actions, self.fc1_nodes, self.enable_dueling_dqn).to(device=device)

        if is_training:
            # Initialize experience replay buffer
            memory = ReplayMemory(self.replay_memory_size, seed=42)
            epsilon = self.epsilon_start

            # Target DQN for stabilizing learning (updated less frequently)
            target_dqn = DQN(num_states, num_actions, self.fc1_nodes, self.enable_dueling_dqn).to(device=device)
            target_dqn.load_state_dict(policy_dqn.state_dict())

            # Optimizer setup
            count = 0
            self.optimizer = pt.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate)
            best_reward = -float('inf')  # For saving best model

        else:
            # Evaluation mode: Load trained model and set to eval()
            policy_dqn.load_state_dict(pt.load(self.MODEL_FILE))
            policy_dqn.eval()

        # Episode loop (runs forever until manually stopped or goal reached)
        for epoc in itertools.count():
            state, _ = env.reset()
            state = pt.tensor(state, dtype=pt.float32, device=device)
            terminated = False
            epoc_reward = 0

            # Step loop: Keep interacting with environment until terminated
            while not terminated and epoc_reward < self.stop_on_reward:
                # Epsilon-greedy action selection
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                    action = pt.tensor(action, dtype=pt.int64, device=device)
                else:
                    with pt.no_grad():
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

                # Apply action and observe outcome
                new_state, reward, terminated, _, info = env.step(action.item())
                new_state = pt.tensor(new_state, dtype=pt.float32, device=device)
                epoc_reward += reward
                reward = pt.tensor(reward, dtype=pt.float, device=device)

                if is_training:
                    # Store experience in replay buffer
                    memory.append((state, action, new_state, reward, terminated))
                    count += 1

                state = new_state  # Move to next state

            rewards_per_epoc.append(epoc_reward)

            # During training: Save model, update graph, train with sampled batch
            if is_training:
                if epoc_reward > best_reward:
                    best_reward = epoc_reward
                    log_message = f"New best reward: {best_reward} at epoc {epoc}"
                    print(log_message)
                    with open(self.LOG_FILE, 'a') as log_file:
                        log_file.write(f"{datetime.now().strftime(DATE_FORMAT)}: {log_message}\n")
                    pt.save(policy_dqn.state_dict(), self.MODEL_FILE)  # Save best model

                # Save graph every 10 seconds
                current_time = datetime.now()
                if current_time - last_graph_update_time > timedelta(seconds=10):
                    self.save_graph(rewards_per_epoc, epsilon_history)
                    last_graph_update_time = current_time

                # Learn from experience replay if there's enough data
                if len(memory) > self.batch_size:
                    batch = memory.sample(self.batch_size)
                    self.optimize(batch, policy_dqn, target_dqn)

                    # Decay exploration rate
                    epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                    epsilon_history.append(epsilon)

                    # Periodically sync target network with policy
                    if count > self.sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        count = 0

    def optimize(self, batch, policy_dqn, target_dqn):
        """
        Perform a single training step using a mini-batch of experiences.
        Implements the Bellman equation for value estimation.
        """
        states, actions, new_states, rewards, terminations = zip(*batch)

        # Stack into tensors
        states = pt.stack(states)
        actions = pt.stack(actions)
        new_states = pt.stack(new_states)
        rewards = pt.tensor(rewards)
        terminations = pt.tensor(terminations).float().to(device)

        with pt.no_grad():
            if self.enable_double_dqn:
                # Double DQN: Select best action from policy, evaluate it using target
                best_actions = policy_dqn(new_states).argmax(dim=1)
                target_q = rewards + (1 - terminations) * self.discount_factor * \
                           target_dqn(new_states).gather(1, best_actions.unsqueeze(1)).squeeze()
            else:
                # Standard DQN: Max Q-value from target
                target_q = rewards + (1 - terminations) * self.discount_factor * \
                           target_dqn(new_states).max(1)[0]

        # Predicted Q-values for taken actions
        current_q = policy_dqn(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Compute loss between target and current predictions
        loss = self.loss_fn(current_q, target_q)

        # Backpropagation step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_graph(self, rewards_per_episode, epsilon_history):
        """
        Save reward and epsilon graphs to visualize learning progress.
        """
        fig = plt.figure(1)

        # Compute moving average of rewards
        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x - 99):(x + 1)])

        # Plot mean reward
        plt.subplot(121)
        plt.ylabel('Mean Rewards')
        plt.plot(mean_rewards)

        # Plot epsilon decay
        plt.subplot(122)
        plt.ylabel('Epsilon Decay')
        plt.plot(epsilon_history)

        plt.subplots_adjust(wspace=1.0, hspace=1.0)
        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)

if __name__ == "__main__":
    # Command-line interface to choose training or evaluation
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('hyperparameters', help='Name of hyperparameter set in YAML file.')
    parser.add_argument('--train', help='Training mode', action='store_true')
    args = parser.parse_args()

    # Create agent with selected hyperparameters
    dql = Agent(hyperparams_set=args.hyperparameters)

    # Train or evaluate based on input
    if args.train:
        dql.run(is_training=True)
    else:
        dql.run(is_training=False, render=True)
