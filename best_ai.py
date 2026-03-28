import os
import torch
import random
import copy
from hexo_engine import HeXOEngine, Hex
from model import HeXONet
from train import NeuralMCTS, device, BOARD_SIZE

class HeXOBestAI:
    def __init__(self, player_id: int, model_path="hexo_model.pth"):
        self.player_id = player_id
        self.use_neural = os.path.exists(model_path)
        
        if self.use_neural:
            print(f"Loading best model for Player {player_id} from {model_path}...")
            self.model = HeXONet(board_size=BOARD_SIZE).to(device)
            # Add weights_only=True for safe loading, or just ignore for now
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            self.model.eval()
            self.mcts = NeuralMCTS(self.model)
        else:
            print(f"No model found. Player {player_id} will use Random AI.")

    def choose_move(self, engine: HeXOEngine) -> list[Hex]:
        moves_needed = engine.get_moves_allowed() - engine.moves_made_this_turn
        moves = []
        
        sim_engine = copy.deepcopy(engine)
        
        for _ in range(moves_needed):
            if self.use_neural:
                # Use temp=0 to always pick the most simulated move (greedy best play)
                pi, legal_moves = self.mcts.getActionProb(sim_engine, temp=0)
                try:
                    best_idx = pi.index(1.0)
                    best_move = legal_moves[best_idx]
                except ValueError:
                    # Fallback if probability array doesn't have 1.0
                    best_move = random.choice(legal_moves)
            else:
                legal = sim_engine.get_legal_moves()
                if not legal:
                    break
                best_move = random.choice(legal)
                
            moves.append(best_move)
            sim_engine.place_stone(best_move)
            
        return moves
