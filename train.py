import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from model import HeXONet
from hexo_engine import HeXOEngine, Hex
import random
import time
import math
import numpy as np

# Training Constants optimized for Ada GPUs
BATCH_SIZE = 128
EPOCHS = 10
BOARD_SIZE = 21 # Accommodates max radius of ~8+ across all directions
SIMULATIONS = 50 # MCTS simulations per turn
GAMES = 20 # Self-play games per epoch

device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True # Optimized for RTX Ada
    torch.set_float32_matmul_precision('high') # TF32 optimization for RTX Ada

def encode_board(engine: HeXOEngine, player_id: int):
    # Center (0,0) mapped to (10, 10). Wait, q and r can range more if stones expand 
    # but the max distance is tightly controlled by the engine if the board starts at 0,0.
    # We will compute dynamic bounding box, but for fully fixed tensor we pad to 21x21.
    matrix = np.zeros((2, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    center = BOARD_SIZE // 2
    for h, p in engine.board.items():
        q_idx = h.q + center
        r_idx = h.r + center
        if 0 <= q_idx < BOARD_SIZE and 0 <= r_idx < BOARD_SIZE:
            layer = 0 if p == player_id else 1
            matrix[layer, q_idx, r_idx] = 1.0
    return matrix

def hex_to_idx(h: Hex):
    center = BOARD_SIZE // 2
    return (h.q + center) * BOARD_SIZE + (h.r + center)

def idx_to_hex(idx: int):
    center = BOARD_SIZE // 2
    q = idx // BOARD_SIZE - center
    r = idx % BOARD_SIZE - center
    return Hex(q, r)

class HeXODataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        board, pi, z = self.data[idx]
        return torch.tensor(board, dtype=torch.float32), torch.tensor(pi, dtype=torch.float32), torch.tensor(z, dtype=torch.float32)

class NeuralMCTS:
    def __init__(self, model, c_puct=1.5):
        self.model = model
        self.c_puct = c_puct
        self.Qsa = {}  
        self.Nsa = {}  
        self.Ns = {}   
        self.Ps = {}   
        self.Es = {}
        self.Vs = {}

    def getActionProb(self, state: HeXOEngine, temp=1):
        # We need a string representation of the state
        # In HeXO, stones and current player define the state
        s = str(sorted(state.board.items())) + str(state.current_player)
        
        for _ in range(SIMULATIONS):
             self.search(state)
             
        legal_moves = state.get_legal_moves()
        counts = [self.Nsa.get((s, m), 0) for m in legal_moves]
        
        if sum(counts) == 0:
            # Fallback if unexpanded
            probs = [1.0/len(legal_moves)] * len(legal_moves)
        elif temp == 0:
            bestAs = np.argwhere(counts == np.max(counts)).flatten()
            bestA = random.choice(bestAs)
            probs = [0] * len(legal_moves)
            probs[bestA] = 1
        else:
            counts = [x ** (1. / temp) for x in counts]
            counts_sum = float(sum(counts))
            probs = [x / counts_sum for x in counts]
            
        return probs, legal_moves

    def search(self, state: HeXOEngine):
        import copy
        s = str(sorted(state.board.items())) + str(state.current_player)
        
        if s not in self.Es:
            self.Es[s] = self._check_terminal(state)
        if self.Es[s] != 0:
            return -self.Es[s]

        if s not in self.Ps:
            # Leaf node expansion using Neural Net
            board_tensor = torch.tensor(encode_board(state, state.current_player)).unsqueeze(0).to(device)
            with torch.no_grad():
                with torch.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                    pi, v = self.model(board_tensor)
                
            pi = torch.exp(pi).cpu().numpy()[0]
            
            legal_moves = state.get_legal_moves()
            self.Vs[s] = legal_moves
            
            # Mask illegal moves
            legal_idx = [hex_to_idx(m) for m in legal_moves]
            valid_pi = np.zeros(BOARD_SIZE * BOARD_SIZE)
            for m, idx in zip(legal_moves, legal_idx):
                if 0 <= idx < BOARD_SIZE * BOARD_SIZE:
                    valid_pi[idx] = pi[idx]
            
            sum_Ps_s = np.sum(valid_pi)
            if sum_Ps_s > 0:
                valid_pi /= sum_Ps_s
            else:
                # Fallback purely random
                for idx in legal_idx:
                     if 0 <= idx < len(valid_pi): valid_pi[idx] = 1.0 / len(legal_idx)
            
            self.Ps[s] = valid_pi
            self.Ns[s] = 0
            return -v.item()

        legal_moves = self.Vs[s]
        cur_best = -float('inf')
        best_act = None
        
        for a in legal_moves:
            idx = hex_to_idx(a)
            if idx < 0 or idx >= BOARD_SIZE * BOARD_SIZE: continue
            
            if (s, a) in self.Qsa:
                u = self.Qsa[(s, a)] + self.c_puct * self.Ps[s][idx] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
            else:
                u = self.c_puct * self.Ps[s][idx] * math.sqrt(self.Ns[s] + 1e-8)
                
            if u > cur_best:
                cur_best = u
                best_act = a
                
        if best_act is None: best_act = random.choice(legal_moves)

        next_s = copy.deepcopy(state)
        next_s.place_stone(best_act)

        v = self.search(next_s)

        if (s, best_act) in self.Qsa:
            self.Qsa[(s, best_act)] = (self.Nsa[(s, best_act)] * self.Qsa[(s, best_act)] + v) / (self.Nsa[(s, best_act)] + 1)
            self.Nsa[(s, best_act)] += 1
        else:
            self.Qsa[(s, best_act)] = v
            self.Nsa[(s, best_act)] = 1

        self.Ns[s] += 1
        return -v

    def _check_terminal(self, state):
        if state.game_over:
            # We return positive score from the winning player's perspective.
            # However this function returns terminal value from CURRENT player's perspective.
            # But the final move was played by the PREVIOUS player.
            # So if game is over, the current player MUST have lost!
            return -1
        return 0

def execute_episode(model):
    train_examples = []
    state = HeXOEngine()
    mcts = NeuralMCTS(model)
    
    while True:
        temp = int(state.turn_number < 15)
        pi, moves = mcts.getActionProb(state, temp=temp)
        
        pi_target = np.zeros(BOARD_SIZE * BOARD_SIZE)
        for p, m in zip(pi, moves):
            idx = hex_to_idx(m)
            if 0 <= idx < BOARD_SIZE * BOARD_SIZE:
                pi_target[idx] = p
                
        train_examples.append([encode_board(state, state.current_player), state.current_player, pi_target])
        
        idx = np.random.choice(len(moves), p=pi)
        action = moves[idx]
        state.place_stone(action)
        
        if state.game_over:
            # Reconstruct targets corresponding to final game outcome
            r = []
            for e in train_examples:
                z = 1 if e[1] == state.winner else -1
                r.append((e[0], e[2], z))
            return r

def train_network():
    print(f"Initializing Neural Network on device: {device}")
    model = HeXONet(board_size=BOARD_SIZE).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    for epoch in range(EPOCHS):
        print(f"--- Epoch {epoch+1}/{EPOCHS} ---")
        model.eval()
        train_data = []
        for i in range(GAMES):
            print(f"Self-play Game {i+1}/{GAMES}...")
            train_data += execute_episode(model)
            
        dataset = HeXODataset(train_data)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
        
        model.train()
        total_loss = 0
        
        for boards, pis, zs in dataloader:
            boards, pis, zs = boards.to(device), pis.to(device), zs.to(device)
            optimizer.zero_grad()
            
            if device.type == 'cuda':
                with torch.autocast(device_type='cuda'):
                    out_pi, out_v = model(boards)
                    loss_pi = -torch.sum(pis * F.log_softmax(out_pi, dim=1)) / boards.size(0)
                    loss_v = F.mse_loss(out_v.squeeze(-1), zs)
                    loss = loss_pi + loss_v
                    
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                 out_pi, out_v = model(boards)
                 loss_pi = -torch.sum(pis * F.log_softmax(out_pi, dim=1)) / boards.size(0)
                 loss_v = F.mse_loss(out_v.squeeze(-1), zs)
                 loss = loss_pi + loss_v
                 loss.backward()
                 optimizer.step()
                 
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1} Average Loss: {total_loss/len(dataloader):.4f}")
        torch.save(model.state_dict(), "hexo_model.pth")
        
    print("Training complete! Model saved to hexo_model.pth")

if __name__ == "__main__":
    train_network()
