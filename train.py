import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from model import HeXONet
from hexo_engine import HeXOEngine, Hex
import random
import time
import math
import numpy as np

# Training Constants optimized for high-end Dell Precision (i7-13800H + RTX Ada)
BATCH_SIZE = 256  # Larger batch for better GPU utilization
EPOCHS = 20       # More epochs for better convergence
BOARD_SIZE = 21   
SIMULATIONS = 160 # Deeper MCTS for higher quality data
GAMES = 60        # More games per epoch
NUM_WORKERS = 8   # Parallel self-play processes (utilize those 20 threads!)

device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True 
    torch.set_float32_matmul_precision('high') 

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
        s = hash((frozenset(state.board.items()), state.current_player))
        
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
        s = hash((frozenset(state.board.items()), state.current_player))
        
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

        next_s = state.clone()
        success = next_s.place_stone(best_act)
        if not success:
             # Safety fallback: Illegal moves should be drastically penalized during MCTS loop
             return -100

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

def worker_execute_episode(weights_path):
    # Re-initialize for worker subprocess
    worker_device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    worker_model = HeXONet(board_size=BOARD_SIZE).to(worker_device)
    worker_model.load_state_dict(torch.load(weights_path, map_location=worker_device, weights_only=True))
    worker_model.eval()
    
    train_examples = []
    state = HeXOEngine()
    state.place_stone(Hex(0, 0)) # First move forced
    mcts = NeuralMCTS(worker_model)
    
    while True:
        # We don't need UI progress bars in individual workers
        temp = int(state.turn_number < 15)
        pi, moves = mcts.getActionProb(state, temp=temp)
        
        pi_target = np.zeros(BOARD_SIZE * BOARD_SIZE)
        for p, m in zip(pi, moves):
            idx = (m.q + BOARD_SIZE // 2) * BOARD_SIZE + (m.r + BOARD_SIZE // 2)
            if 0 <= idx < BOARD_SIZE * BOARD_SIZE:
                pi_target[idx] = p
                
        train_examples.append([encode_board(state, state.current_player), state.current_player, pi_target])
        
        idx = np.random.choice(len(moves), p=pi)
        action = moves[idx]
        state.place_stone(action)
        
        if state.game_over:
            r = []
            for e in train_examples:
                z = 1 if e[1] == state.winner else -1
                r.append((e[0], e[2], z))
            return r, len(train_examples)
            
        if len(train_examples) >= 200: # Limit early games
            r = []
            for e in train_examples:
                r.append((e[0], e[2], 0))
            return r, len(train_examples)

def execute_episode(model, pbar, start_t):
    train_examples = []
    state = HeXOEngine()
    
    # Auto-play the deterministically forced first move to prevent useless 0-state NN evaluations
    state.place_stone(Hex(0, 0))
    
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
        
        # Real-time UI Telemetry Update
        elapsed = time.time() - start_t
        current_moves = len(train_examples)
        sps = (current_moves * SIMULATIONS) / elapsed if elapsed > 0 else 0
        pbar.set_postfix_str(f"moves={current_moves} sps={sps:.1f}")
        
        if state.game_over:
            # Reconstruct targets corresponding to final game outcome
            r = []
            for e in train_examples:
                z = 1 if e[1] == state.winner else -1
                r.append((e[0], e[2], z))
            return r, len(train_examples)
            
        if len(train_examples) >= 150:
            # Prevent infinite random-walk games without a winner during early untrained epochs
            r = []
            for e in train_examples:
                r.append((e[0], e[2], 0))  # Draw = 0 reward
            return r, len(train_examples)

def train_network():
    print(f"HeXO Training — Device: {device}")
    print(f"Config: {GAMES} games × {SIMULATIONS} sims, batch={BATCH_SIZE}")
    print()
    
    model = HeXONet(board_size=BOARD_SIZE).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    print("-" * 50)
    print("MODEL DIAGNOSTICS:")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Board Tensors: {BOARD_SIZE}x{BOARD_SIZE}")
    print(f"Hidden Dimensions: 128")
    print(f"Number of CNN Layers: 3")
    print(f"Total Parameters: {total_params:,}")
    print(f"Model Device Placement: {device}")
    print("-" * 50)
    print()
    
    from tqdm import tqdm
    import concurrent.futures
    import multiprocessing
    
    # We use 'spawn' or 'fork' based on OS, but for CUDA must use 'spawn'
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    for epoch in range(EPOCHS):
        print(f"── Epoch {epoch+1}/{EPOCHS} ──")
        model.eval()
        train_data = []
        
        print(f"    Generating {GAMES} games across {NUM_WORKERS} workers...")
        
        # Parallel game generation
        with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            # We must pass a path or a state_dict to reconstruct the model in each process
            # since a full torch Model can't always be pickled across processes cleanly with CUDA.
            # However, for simplicity here we pass the current model if it works, 
            # or just load it inside each process. 
            # A more robust way is to save a temp weight file.
            torch.save(model.state_dict(), "temp_model_sync.pth")
            
            futures = []
            results_count = 0
            pbar = tqdm(total=GAMES, desc="    Total Progress")
            
            # Start games
            for _ in range(GAMES):
                futures.append(executor.submit(worker_execute_episode, "temp_model_sync.pth"))
            
            for future in concurrent.futures.as_completed(futures):
                data, moves = future.result()
                train_data += data
                results_count += 1
                pbar.update(1)
            pbar.close()

        print(f"    Self-play complete. Generated {len(train_data)} training examples.")
            
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
