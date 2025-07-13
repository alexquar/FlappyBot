import torch as pt 
from torch import nn 
import torch.nn.functional as f


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.out = nn.Linear(256, output_dim)

    def forward(self, x):
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.out(x)
        return x
    
    
def main():
    # Example usage
    input_dim = 12  # Example input dimension (e.g., state size)
    output_dim = 2  # Example output dimension (e.g., action size)
    
    model = DQN(input_dim, output_dim)
    
    # Example input tensor
    x = pt.randn(1, input_dim)
    
    # Forward pass
    output = model(x)
    print(output)
    
if __name__ == "__main__":
    main()