from collections import deque
import random

class ReplayMemory():
    """ A simple replay memory to store transitions for reinforcement learning agents."""
    def __init__(self, maxlen, seed=None):
        """ 
        Initialize the replay memory.
        Args:
            maxlen (int): Maximum number of transitions to store in memory.
            seed (int, optional): Random seed for reproducibility. Defaults to None.
        """
        self.memory = deque([], maxlen = maxlen)
        if seed is not None:
            random.seed(seed)
            
    def append(self, transition):
        """
        Append a new transition to the memory.
        Args:
            transition (tuple): A transition consisting of (state, action, reward, next_state, done).
        """
        self.memory.append(transition)
    
    def sample(self, batch_size):
        """
        Sample a batch of transitions from the memory.
        Args:
            batch_size (int): Number of transitions to sample.
        """
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        """
        Get the current size of the memory.
        """
        return len(self.memory)