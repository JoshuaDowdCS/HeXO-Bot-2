import os
import torch
import random
from hexo_engine import HeXOEngine, Hex
from model import HeXOMlpNet
from train import NeuralMCTS, device, INPUT_RADIUS, NUM_GLOBAL_FEATURES
from ai import HeXOAI

class HeXOBestAI:
    def __init__(self, player_id: int, model_path="hexo_mlp_model.pth"):
        self.player_id = player_id
        self.use_neural = os.path.exists(model_path)

        if self.use_neural:
            print(f"Loading best model for Player {player_id} from {model_path}...")
            self.model = HeXOMlpNet(input_radius=INPUT_RADIUS, num_global_features=NUM_GLOBAL_FEATURES).to(device)
            self.model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            self.model.eval()
            self.mcts = NeuralMCTS(self.model)
        else:
            print(f"No model found. Player {player_id} will use Heuristic AI fallback.")
            self.fallback_ai = HeXOAI(player_id)

    def choose_move(self, engine: HeXOEngine) -> list[Hex]:
        if not self.use_neural:
            return self.fallback_ai.choose_move(engine)

        moves_needed = engine.get_moves_allowed() - engine.moves_made_this_turn
        moves = []

        sim_engine = engine.clone()

        for _ in range(moves_needed):
            pi, legal_moves = self.mcts.getActionProb(sim_engine, temp=0)
            if not legal_moves:
                break
            try:
                best_idx = pi.index(1.0)
                best_move = legal_moves[best_idx]
            except ValueError:
                # Fallback if probability array doesn't have a clean 1.0
                best_move = random.choice(legal_moves)

            moves.append(best_move)
            sim_engine.place_stone(best_move)

        return moves
