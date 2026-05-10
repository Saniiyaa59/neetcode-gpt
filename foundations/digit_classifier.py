import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.layer1 = nn.Linear(784, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.layer2 = nn.Linear(512, 10)
        self.sigmoid = nn.Sigmoid()
        
        # Define the architecture here
    
    def forward(self, images: TensorType[float]) -> TensorType[float]:
        torch.manual_seed(0)
        N = images.size(0)
        input_ = images.view(N, -1)
        x = self.layer1(input_)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.sigmoid(x)
        return x
        # Return the model's prediction to 4 decimal places