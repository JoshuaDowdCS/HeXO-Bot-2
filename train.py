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
# 31x31 allows for radius 15 sight, covering most HeXO games easily.
BATCH_SIZE = 256  
EPOCHS = 20       
BOARD_SIZE = 31   
SIMULATIONS = 200 # Higher quality MCTS to learn defensive structures
GAMES = 60        
NUM_WORKERS = 16  # Parallel self-play (maximizes 20 threads)
BOOTSTRAP_GAMES = 10 # Initial games from Heuristic AI to avoid starting from zero

device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True 
    torch.set_float32_matmul_precision('high') 

def get_board_centroid(engine: HeXOEngine):
    if not engine.board:
         return 0, 0
    qs = [h.q for h in engine.board]
    rs = [h.r for h in engine.board]
    return int(round(np.mean(qs))), int(round(np.mean(rs)))

def encode_board(engine: HeXOEngine, player_id: int):
    # Centroid-based sliding window: map board center to (10, 10)
    matrix = np.zeros((2, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    center = BOARD_SIZE // 2
    offset_q, offset_r = get_board_centroid(engine)
    
    for h, p in engine.board.items():
        q_idx = h.q - offset_q + center
        r_idx = h.r - offset_r + center
        if 0 <= q_idx < BOARD_SIZE and 0 <= r_idx < BOARD_SIZE:
            layer = 0 if p == player_id else 1
            matrix[layer, q_idx, r_idx] = 1.0
    return matrix

def hex_to_idx(h: Hex, offset_q: int, offset_r: int):
    center = BOARD_SIZE // 2
    q_idx = h.q - offset_q + center
    r_idx = h.r - offset_r + center
    if 0 <= q_idx < BOARD_SIZE and 0 <= r_idx < BOARD_SIZE:
        return q_idx * BOARD_SIZE + r_idx
    return None

def idx_to_hex(idx: int, offset_q: int, offset_r: int):
    center = BOARD_SIZE // 2
    q = idx // BOARD_SIZE - center + offset_q
    r = idx % BOARD_SIZE - center + offset_r
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
        s = state.get_state_key()
        
        for _ in range(SIMULATIONS):
             self.search(state)
             
        legal_moves = state.get_legal_moves()
        counts = [self.Nsa.get((s, m), 0) for m in legal_moves]
        
        if len(legal_moves) == 0:
            return [], []
            
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
        s = state.get_state_key()
        
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
                
            pi = torch.softmax(pi, dim=1).cpu().numpy()[0]
            
            legal_moves = state.get_legal_moves()
            if not legal_moves:
                self.Es[s] = -0.1 # Draw/Invalid
                return 0.1

            self.Vs[s] = legal_moves
            offset_q, offset_r = get_board_centroid(state)
            
            # Mask illegal moves
            valid_pi = np.zeros(BOARD_SIZE * BOARD_SIZE)
            for m in legal_moves:
                idx = hex_to_idx(m, offset_q, offset_r)
                if idx is not None:
                    valid_pi[idx] = pi[idx]
            
            sum_Ps_s = np.sum(valid_pi)
            if sum_Ps_s > 1e-9:
                valid_pi /= sum_Ps_s
            else:
                # Fallback: uniform over legal moves in sight
                in_sight = 0
                for m in legal_moves:
                    idx = hex_to_idx(m, offset_q, offset_r)
                    if idx is not None:
                        valid_pi[idx] = 1.0
                        in_sight += 1
                if in_sight > 0:
                    valid_pi /= in_sight
                else:
                    # Rare: all moves out of sight, just allow expansion
                    pass
            
            self.Ps[s] = valid_pi
            self.Ns[s] = 0
            return -v.item()

        legal_moves = self.Vs[s]
        offset_q, offset_r = get_board_centroid(state)
        
        # Ps was stored as valid_pi (array of size 441)
        # Filter out moves that are out of bounds for the NN's current sliding window
        valid_legal_moves = []
        indices = []
        for a in legal_moves:
            idx = hex_to_idx(a, offset_q, offset_r)
            if idx is not None:
                valid_legal_moves.append(a)
                indices.append(idx)
        
        if not valid_legal_moves:
            # If NO moves are in sight, pick the one closest to centroid
            dists = [Hex(a.q - offset_q, a.r - offset_r).distance(Hex(0,0)) for a in legal_moves]
            best_act = legal_moves[np.argmin(dists)]
            # We don't search further, just take it
            return -0.1
            
        ps_vals = self.Ps[s][indices] 
        counts = np.array([self.Nsa.get((s, a), 0) for a in valid_legal_moves])
        q_vals = np.array([self.Qsa.get((s, a), 0) for a in valid_legal_moves])
        
        u = q_vals + self.c_puct * ps_vals * (math.sqrt(self.Ns[s] + 1e-8) / (1 + counts))
        best_idx = np.argmax(u)
        best_act = valid_legal_moves[best_idx]

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

def worker_execute_episode(weights_path, shared_moves=None, shared_games=None):
    # Re-initialize for worker subprocess
    worker_device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    worker_model = HeXONet(board_size=BOARD_SIZE).to(worker_device)
    worker_model.load_state_dict(torch.load(weights_path, map_location=worker_device, weights_only=True))
    worker_model.eval()
    
    train_examples = []
    state = HeXOEngine(boundary_radius=BOARD_SIZE // 2)
    state.place_stone(Hex(0, 0)) # First move forced
    mcts = NeuralMCTS(worker_model)
    
    while True:
        temp = int(state.turn_number < 15)
        pi, moves = mcts.getActionProb(state, temp=temp)
        
        offset_q, offset_r = get_board_centroid(state)
        pi_target = np.zeros(BOARD_SIZE * BOARD_SIZE)
        for p, m in zip(pi, moves):
            idx = hex_to_idx(m, offset_q, offset_r)
            if idx is not None:
                pi_target[idx] = p
                
        train_examples.append([encode_board(state, state.current_player), state.current_player, pi_target])
        if shared_moves:
             shared_moves.value += 1
        
        if not moves:
             break

        idx = np.random.choice(len(moves), p=pi)
        action = moves[idx]
        state.place_stone(action)
        
        if state.game_over:
            r = []
            for e in train_examples:
                z = 1 if e[1] == state.winner else -1
                r.append((e[0], e[2], z))
            if shared_games: shared_games.value += 1
            return r, len(train_examples)
            
        if len(train_examples) >= 200: # Limit early games
            r = []
            for e in train_examples:
                r.append((e[0], e[2], 0))
            if shared_games: shared_games.value += 1
            return r, len(train_examples)

def execute_episode(model, pbar, start_t):
    train_examples = []
    state = HeXOEngine(boundary_radius=BOARD_SIZE // 2)
    
    # Auto-play the deterministically forced first move to prevent useless 0-state NN evaluations
    state.place_stone(Hex(0, 0))
    
    mcts = NeuralMCTS(model)
    
    while True:
        temp = int(state.turn_number < 15)
        pi, moves = mcts.getActionProb(state, temp=temp)
        
        offset_q, offset_r = get_board_centroid(state)
        pi_target = np.zeros(BOARD_SIZE * BOARD_SIZE)
        for p, m in zip(pi, moves):
            idx = hex_to_idx(m, offset_q, offset_r)
            if idx is not None:
                pi_target[idx] = p
                
        train_examples.append([encode_board(state, state.current_player), state.current_player, pi_target])
        
        if not moves:
             break

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

def _worker_bootstrap_episode(game_idx):
    from ai import HeXOAI
    state = HeXOEngine()
    state.place_stone(Hex(0, 0))
    h_ai = {1: HeXOAI(1), 2: HeXOAI(2)}
    game_examples = []
    
    # 0.1s is enough for depth-4 search which is plenty for decent bootstrapping
    while not state.game_over and len(game_examples) < 100:
         moves = h_ai[state.current_player].choose_move(state, time_limit=0.1)
         
         # Create a target policy from heuristic move
         pi_target = np.zeros(BOARD_SIZE * BOARD_SIZE)
         offset_q, offset_r = get_board_centroid(state)
         for m in moves:
              idx = hex_to_idx(m, offset_q, offset_r)
              if idx is not None:
                   pi_target[idx] = 1.0 / len(moves)
         
         game_examples.append([encode_board(state, state.current_player), state.current_player, pi_target])
         for m in moves: state.place_stone(m)
         
    # Value targets
    train_data = []
    for e in game_examples:
        z = 1 if e[1] == state.winner else (-1 if state.winner is not None else 0)
        train_data.append((e[0], e[2], z))
    return train_data, len(game_examples)

def bootstrap_with_heuristic(num_games=10):
    import concurrent.futures
    from tqdm import tqdm
    print(f"    [BOOTSTRAPPING] Generating {num_games} games in parallel...")
    train_data = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(num_games, NUM_WORKERS)) as executor:
        futures = [executor.submit(_worker_bootstrap_episode, i) for i in range(num_games)]
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            data, moves_count = future.result()
            train_data += data
            print(f"      Game {i+1}/{num_games} done ({moves_count} moves)")
            
    return train_data

def train_network():
    print(f"HeXO Training — Device: {device}")
    print(f"Config: {GAMES} games × {SIMULATIONS} sims, batch={BATCH_SIZE}")
    print()
    
    model = HeXONet(board_size=BOARD_SIZE).to(device)
    if os.path.exists("hexo_model.pth"):
        print(f"Loading existing model from hexo_model.pth...")
        model.load_state_dict(torch.load("hexo_model.pth", map_location=device, weights_only=True))
    
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
    from multiprocessing import Manager
    import multiprocessing
    import concurrent.futures
    
    # We use 'spawn' or 'fork' based on OS, but for CUDA must use 'spawn'
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    for epoch in range(EPOCHS):
        print(f"── Epoch {epoch+1}/{EPOCHS} ──")
        
        # Heuristic Bootstrapping for Epoch 1 only
        if epoch == 0 and BOOTSTRAP_GAMES > 0:
            train_data = bootstrap_with_heuristic(BOOTSTRAP_GAMES)
            print(f"    Bootstrap complete. Playing self-play games to refine...")
        else:
            train_data = []
        
        model.eval()
        
        print(f"    Generating {GAMES} games across {NUM_WORKERS} workers...")
        
        with Manager() as manager:
            shared_moves = manager.Value('i', 0)
            shared_games = manager.Value('i', 0)
            
            torch.save(model.state_dict(), "temp_model_sync.pth")
            
            start_t = time.time()
            pbar = tqdm(total=GAMES, desc="    Total Progress")

            with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
                futures = [executor.submit(worker_execute_episode, "temp_model_sync.pth", shared_moves, shared_games) for _ in range(GAMES)]
                
                while True:
                    results_count = sum(1 for f in futures if f.done())
                    
                    # Update UI postfix with global stats
                    elapsed = time.time() - start_t
                    total_moves = shared_moves.value
                    global_sps = (total_moves * SIMULATIONS) / elapsed if elapsed > 0 else 0
                    
                    # Update bar only as new games finish or during gaps
                    pbar.n = results_count
                    pbar.set_postfix_str(f"total_moves={total_moves} global_sps={global_sps:.1f}")
                    pbar.refresh()
                    
                    if results_count == GAMES:
                        break
                    time.sleep(1) # Refresh rate 1Hz

                for future in concurrent.futures.as_completed(futures):
                    data, _ = future.result()
                    train_data += data
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
