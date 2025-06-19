# dqn_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()

        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Bereken de grootte van de input voor de lineaire laag
        # Although this function is defined, it is not used for size calculation.
        # The dummy input approach is correct for determining the size.
        def conv2d_size_out(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride  + 1

        # Dummy input om de outputgrootte van de conv-lagen te bepalen
        dummy_input = torch.zeros(1, *input_shape)
        # Normalize dummy input as done in forward pass
        dummy_input = dummy_input / 255.0
        conv_out = self.conv3(self.conv2(self.conv1(dummy_input)))
        # Use reshape instead of view for determining flattened size for consistency and robustness
        flattened_size = conv_out.reshape(1, -1).size(1)

        self.fc1 = nn.Linear(flattened_size, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        # Normaliseer de input
        x = x / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # Use reshape instead of view for flattening the tensor
        x = x.reshape(x.size(0), -1) # Flatten
        x = F.relu(self.fc1(x))
        return self.fc2(x)