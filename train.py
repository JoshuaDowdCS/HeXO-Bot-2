import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from model import HeXOMlpNet, build_hex_grid
from hexo_engine import HeXOEngine, Hex
import random
import time
import math
import numpy as np

# ── Training Constants optimized for RTX 2000 Ada ──────────────────────
BATCH_SIZE = 256
EPOCHS = 20
INPUT_RADIUS = 15       # Hex sight radius (covers 30 hexes across)
NUM_GLOBAL_FEATURES = 6  # Global awareness features
SIMULATIONS = 200
GAMES = 60
NUM_WORKERS = 16
BOOTSTRAP_GAMES = 10
REPLAY_MEMORY_SIZE = 500

# ── Precompute hex grid (deterministic ordering) ───────────────────────
GRID_CELLS = build_hex_grid(INPUT_RADIUS)
NUM_CELLS = len(GRID_CELLS)  # 3*R^2 + 3*R + 1 = 721 for R=15
CELL_TO_IDX = {(q, r): i for i, (q, r) in enumerate(GRID_CELLS)}

# Precompute coordinate template (q/R and r/R never change, only occupancy does)
_COORD_TEMPLATE = np.zeros(NUM_CELLS * 3 + NUM_GLOBAL_FEATURES, dtype=np.float32)
for _i, (_q, _r) in enumerate(GRID_CELLS):
    _COORD_TEMPLATE[_i * 3] = _q / INPUT_RADIUS
    _COORD_TEMPLATE[_i * 3 + 1] = _r / INPUT_RADIUS

# ── Device Setup ───────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')


# ── Encoding Functions ─────────────────────────────────────────────────
def encode_board_flat(engine: HeXOEngine, player_id: int):
    """Sparse flat encoding: [q/R, r/R, occupancy] per hex cell + global features.
    Only iterates over occupied cells (O(num_stones)), not entire grid."""
    features = _COORD_TEMPLATE.copy()
    offset_q, offset_r = engine.get_centroid()

    own_outside = 0
    opp_outside = 0

    for h, p in engine.board.items():
        rel_q = h.q - offset_q
        rel_r = h.r - offset_r
        idx = CELL_TO_IDX.get((rel_q, rel_r))
        if idx is not None:
            features[idx * 3 + 2] = 1.0 if p == player_id else -1.0
        else:
            if p == player_id:
                own_outside += 1
            else:
                opp_outside += 1

    # Global features — gives the model awareness beyond the sight radius
    base = NUM_CELLS * 3
    features[base]     = own_outside / 50.0        # Own stones outside sight
    features[base + 1] = opp_outside / 50.0        # Opponent stones outside sight
    features[base + 2] = len(engine.board) / 100.0  # Game phase (total stones)
    features[base + 3] = engine.turn_number / 100.0  # Turn phase
    features[base + 4] = engine.moves_made_this_turn / 2.0  # Placement within turn
    features[base + 5] = engine.get_moves_allowed() / 2.0   # Moves allowed this turn

    return features


def hex_to_cell_idx(h: Hex, offset_q: int, offset_r: int):
    """Map a hex coordinate to a cell index in the hex grid (centroid-relative)."""
    return CELL_TO_IDX.get((h.q - offset_q, h.r - offset_r))


def cell_idx_to_hex(idx: int, offset_q: int, offset_r: int):
    """Map a cell index back to an absolute hex coordinate."""
    q, r = GRID_CELLS[idx]
    return Hex(q + offset_q, r + offset_r)


# ── Dataset ────────────────────────────────────────────────────────────
class HeXODataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        board, pi, z = self.data[idx]
        return torch.tensor(board, dtype=torch.float32), torch.tensor(pi, dtype=torch.float32), torch.tensor(z, dtype=torch.float32)


# ── Neural MCTS ────────────────────────────────────────────────────────
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
        self.Ps_indices = {}

    def getActionProb(self, state: HeXOEngine, temp=1):
        s = state.get_state_key()

        for _ in range(SIMULATIONS):
            self.search(state)

        legal_moves = state.get_legal_moves()
        counts = [self.Nsa.get((s, m), 0) for m in legal_moves]

        if len(legal_moves) == 0:
            return [], []

        if sum(counts) == 0:
            probs = [1.0 / len(legal_moves)] * len(legal_moves)
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
            # ── Leaf node: expand with NN ──
            board_tensor = torch.tensor(encode_board_flat(state, state.current_player)).unsqueeze(0).to(device)
            with torch.no_grad():
                with torch.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                    pi, v = self.model(board_tensor)

            pi = torch.softmax(pi, dim=1).cpu().numpy()[0]

            legal_moves = state.get_legal_moves()
            if not legal_moves:
                self.Es[s] = -0.1
                return 0.1

            self.Vs[s] = legal_moves
            offset_q, offset_r = state.get_centroid()

            valid_pi = np.zeros(NUM_CELLS)
            valid_moves = []
            valid_indices = []

            for m in legal_moves:
                idx = hex_to_cell_idx(m, offset_q, offset_r)
                if idx is not None:
                    valid_pi[idx] = pi[idx]
                    valid_moves.append(m)
                    valid_indices.append(idx)

            self.Ps_indices[s] = (valid_moves, valid_indices)

            sum_Ps_s = np.sum(valid_pi)
            if sum_Ps_s > 1e-9:
                valid_pi /= sum_Ps_s
            else:
                in_sight = 0
                for m, idx in zip(*self.Ps_indices[s]):
                    valid_pi[idx] = 1.0
                    in_sight += 1
                if in_sight > 0:
                    valid_pi /= in_sight

            self.Ps[s] = valid_pi
            self.Ns[s] = 0
            return -v.item()

        # ── Internal node: select by PUCT ──
        valid_legal_moves, indices = self.Ps_indices[s]

        if not valid_legal_moves:
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
            return -1
        return 0


# ── Self-Play Workers ──────────────────────────────────────────────────
def worker_execute_episode(weights_path, shared_moves=None, shared_games=None):
    worker_device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    worker_model = HeXOMlpNet(input_radius=INPUT_RADIUS, num_global_features=NUM_GLOBAL_FEATURES).to(worker_device)
    worker_model.load_state_dict(torch.load(weights_path, map_location=worker_device, weights_only=True))
    worker_model.eval()

    train_examples = []
    state = HeXOEngine(boundary_radius=INPUT_RADIUS)
    state.place_stone(Hex(0, 0))
    mcts = NeuralMCTS(worker_model)

    while True:
        temp = int(state.turn_number < 15)
        pi, moves = mcts.getActionProb(state, temp=temp)

        offset_q, offset_r = state.get_centroid()
        pi_target = np.zeros(NUM_CELLS)
        for p, m in zip(pi, moves):
            idx = hex_to_cell_idx(m, offset_q, offset_r)
            if idx is not None:
                pi_target[idx] = p

        train_examples.append([encode_board_flat(state, state.current_player), state.current_player, pi_target])
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

        if len(train_examples) >= 200:
            r = []
            for e in train_examples:
                r.append((e[0], e[2], 0))
            if shared_games: shared_games.value += 1
            return r, len(train_examples)


def execute_episode(model, pbar, start_t):
    train_examples = []
    state = HeXOEngine(boundary_radius=INPUT_RADIUS)
    state.place_stone(Hex(0, 0))
    mcts = NeuralMCTS(model)

    while True:
        temp = int(state.turn_number < 15)
        pi, moves = mcts.getActionProb(state, temp=temp)

        offset_q, offset_r = state.get_centroid()
        pi_target = np.zeros(NUM_CELLS)
        for p, m in zip(pi, moves):
            idx = hex_to_cell_idx(m, offset_q, offset_r)
            if idx is not None:
                pi_target[idx] = p

        train_examples.append([encode_board_flat(state, state.current_player), state.current_player, pi_target])

        if not moves:
            break

        idx = np.random.choice(len(moves), p=pi)
        action = moves[idx]
        state.place_stone(action)

        elapsed = time.time() - start_t
        current_moves = len(train_examples)
        sps = (current_moves * SIMULATIONS) / elapsed if elapsed > 0 else 0
        pbar.set_postfix_str(f"moves={current_moves} sps={sps:.1f}")

        if state.game_over:
            r = []
            for e in train_examples:
                z = 1 if e[1] == state.winner else -1
                r.append((e[0], e[2], z))
            return r, len(train_examples)

        if len(train_examples) >= 150:
            r = []
            for e in train_examples:
                r.append((e[0], e[2], 0))
            return r, len(train_examples)


# ── Heuristic Bootstrap ───────────────────────────────────────────────
def _worker_bootstrap_episode(game_idx):
    from ai import HeXOAI
    state = HeXOEngine()
    state.place_stone(Hex(0, 0))
    h_ai = {1: HeXOAI(1), 2: HeXOAI(2)}
    game_examples = []

    while not state.game_over and len(game_examples) < 100:
        moves = h_ai[state.current_player].choose_move(state, time_limit=0.1)

        pi_target = np.zeros(NUM_CELLS)
        offset_q, offset_r = state.get_centroid()
        for m in moves:
            idx = hex_to_cell_idx(m, offset_q, offset_r)
            if idx is not None:
                pi_target[idx] = 1.0 / len(moves)

        game_examples.append([encode_board_flat(state, state.current_player), state.current_player, pi_target])
        for m in moves: state.place_stone(m)

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


# ── Main Training Loop ─────────────────────────────────────────────────
def train_network():
    print(f"HeXO Training (MLP) — Device: {device}")
    print(f"Config: {GAMES} games × {SIMULATIONS} sims, batch={BATCH_SIZE}")
    print(f"Sight: hex radius {INPUT_RADIUS} ({NUM_CELLS} cells, {NUM_CELLS*3+NUM_GLOBAL_FEATURES} input features)")
    print()

    model = HeXOMlpNet(input_radius=INPUT_RADIUS, num_global_features=NUM_GLOBAL_FEATURES).to(device)
    model_path = "hexo_mlp_model.pth"

    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    print("-" * 50)
    print("MODEL DIAGNOSTICS:")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Architecture: HeXOMlpNet (flat vector, no convolutions)")
    print(f"Sight Radius: {INPUT_RADIUS} → {NUM_CELLS} hex cells")
    print(f"Input Size: {NUM_CELLS * 3 + NUM_GLOBAL_FEATURES} features ({NUM_CELLS}×3 cell + {NUM_GLOBAL_FEATURES} global)")
    print(f"Hidden Layers: 512 → 256 → 128")
    print(f"Total Parameters: {total_params:,}")
    print(f"Device: {device}")
    print("-" * 50)
    print()

    from tqdm import tqdm
    from multiprocessing import Manager
    import multiprocessing
    import concurrent.futures

    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    replay_buffer = []

    for epoch in range(EPOCHS):
        print(f"── Epoch {epoch+1}/{EPOCHS} ──")

        new_experience = []

        if epoch == 0 and BOOTSTRAP_GAMES > 0:
            new_experience = bootstrap_with_heuristic(BOOTSTRAP_GAMES)
            print(f"    Bootstrap complete. Adding to memory...")

        model.eval()
        print(f"    Generating {GAMES} games across {NUM_WORKERS} workers...")

        with Manager() as manager:
            shared_moves = manager.Value('i', 0)
            shared_games = manager.Value('i', 0)

            torch.save(model.state_dict(), "temp_mlp_sync.pth")

            start_t = time.time()
            pbar = tqdm(total=GAMES, desc="    Total Progress")

            with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
                futures = [executor.submit(worker_execute_episode, "temp_mlp_sync.pth", shared_moves, shared_games) for _ in range(GAMES)]

                while True:
                    results_count = sum(1 for f in futures if f.done())

                    elapsed = time.time() - start_t
                    total_moves = shared_moves.value
                    global_sps = (total_moves * SIMULATIONS) / elapsed if elapsed > 0 else 0

                    pbar.n = results_count
                    pbar.set_postfix_str(f"total_moves={total_moves} global_sps={global_sps:.1f}")
                    pbar.refresh()

                    if results_count == GAMES:
                        break
                    time.sleep(1)

                for future in concurrent.futures.as_completed(futures):
                    data, _ = future.result()
                    new_experience += data
                pbar.close()

        replay_buffer.extend(new_experience)

        max_buffer_size = REPLAY_MEMORY_SIZE * 170
        if len(replay_buffer) > max_buffer_size:
            replay_buffer = replay_buffer[-max_buffer_size:]

        print(f"    Self-play complete. Epoch added {len(new_experience)} examples. Memory Pool: {len(replay_buffer)}.")

        dataset = HeXODataset(replay_buffer)
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
        torch.save(model.state_dict(), model_path)

    print(f"Training complete! Model saved to {model_path}")

if __name__ == "__main__":
    train_network()
