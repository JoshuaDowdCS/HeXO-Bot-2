import torch
import torch.nn as nn
import torch.nn.functional as F

class HeXONet(nn.Module):
    def __init__(self, board_size=19):
        super(HeXONet, self).__init__()
        # Board size: a radius limit of 8 means max q, r differences around 16.
        # We can map axial coordinates (q, r) to a 2D array of size 19x19 or similar.
        self.board_size = board_size
        
        # Input channels: 1 for current player stones, 1 for opponent stones
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Policy head
        self.pol_conv = nn.Conv2d(128, 2, kernel_size=1)
        self.pol_bn = nn.BatchNorm2d(2)
        self.pol_fc = nn.Linear(2 * board_size * board_size, board_size * board_size)
        
        # Value head
        self.val_conv = nn.Conv2d(128, 1, kernel_size=1)
        self.val_bn = nn.BatchNorm2d(1)
        self.val_fc1 = nn.Linear(board_size * board_size, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # x shape: (batch_size, 2, board_size, board_size)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Policy Head
        pol = F.relu(self.pol_bn(self.pol_conv(x)))
        pol = pol.view(pol.size(0), -1)
        pol = self.pol_fc(pol)
        # Returns logits for coordinates mapped to 1D
        
        # Value Head
        val = F.relu(self.val_bn(self.val_conv(x)))
        val = val.view(val.size(0), -1)
        val = F.relu(self.val_fc1(val))
        val = torch.tanh(self.val_fc2(val))
        
        return pol, val
