import torch.nn as nn
from tqdm import tqdm
import torch


class CombinedColorizationModel(nn.Module):
    """Feed-forward network that will process the concatenated features and colorize an image"""

    def __init__(self, input_size, hidden_size, output_size):
        super(CombinedColorizationModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, n_of_images):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        x = x.view(n_of_images, 32, 32, 3)
        x = x / x.max()
        return x


