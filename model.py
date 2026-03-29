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


def build_hex_grid(radius):
    """Build ordered list of (q,r) cells in a hex grid of given radius.
    This ordering MUST be consistent between training and inference."""
    cells = []
    for q in range(-radius, radius + 1):
        for r in range(max(-radius, -q - radius), min(radius, -q + radius) + 1):
            cells.append((q, r))
    return cells


class HeXOMlpNet(nn.Module):
    def __init__(self, input_radius=15, num_global_features=6):
        super(HeXOMlpNet, self).__init__()
        self.radius = input_radius
        self.num_cells = 3 * input_radius**2 + 3 * input_radius + 1
        input_size = self.num_cells * 3 + num_global_features

        # Shared trunk with dropout for regularization
        self.fc1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)

        # Policy head
        self.pol_fc1 = nn.Linear(128, 128)
        self.pol_fc2 = nn.Linear(128, self.num_cells)

        # Value head
        self.val_fc1 = nn.Linear(128, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = F.relu(self.bn3(self.fc3(x)))

        pol = F.relu(self.pol_fc1(x))
        pol = self.pol_fc2(pol)

        val = F.relu(self.val_fc1(x))
        val = torch.tanh(self.val_fc2(val))

        return pol, val
