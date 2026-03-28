import time
import random
from hexo_engine import HeXOEngine, Hex
from best_ai import HeXOBestAI
from ai import HeXOAI

class RandomAI:
    def __init__(self, player_id):
        self.player_id = player_id

    def choose_move(self, engine: HeXOEngine):
        # Determine how many stones we can place this turn
        moves_needed = engine.get_moves_allowed() - engine.moves_made_this_turn
        moves = []
        
        # We use a clone to simulate the first move before picking the second
        sim_engine = engine.clone()
        
        for _ in range(moves_needed):
            legal = sim_engine.get_legal_moves()
            if not legal:
                break
            move = random.choice(legal)
            moves.append(move)
            sim_engine.place_stone(move)
            
        return moves

def play_game(player1, player2, verbose=False):
    engine = HeXOEngine()
    
    while not engine.game_over:
        current_ai = player1 if engine.current_player == 1 else player2
        
        # Get moves from the AI
        moves = current_ai.choose_move(engine)
        
        for m in moves:
            if not engine.place_stone(m):
                # If AI returns an illegal move, it loses immediately (safety check)
                if verbose: print(f"Player {engine.current_player} played illegal move {m} and forfeits!")
                return 3 - engine.current_player
                
        if verbose:
            print(f"Turn {engine.turn_number-1}: Player {3-engine.current_player} placed {len(moves)} stones. Total stones: {len(engine.board)}")

    if verbose:
        print(f"Game Over! Winner: Player {engine.winner}")
    return engine.winner

def run_benchmark(num_games=10):
    print(f"Starting Benchmark: Trained AI vs. Random AI ({num_games} games)")
    print("-" * 50)
    
    # Initialize AIs
    # Player 1: Trained (uses hexo_model.pth if available)
    # Player 2: Random
    trained_ai = HeXOBestAI(player_id=1)
    random_ai = RandomAI(player_id=2)
    
    results = {1: 0, 2: 0, "draw": 0}
    
    for i in range(num_games):
        # Alternate who goes first every game for fairness
        if i % 2 == 0:
            p1, p2 = trained_ai, random_ai
            trained_id, random_id = 1, 2
        else:
            p1, p2 = random_ai, trained_ai
            trained_id, random_id = 2, 1
            
        print(f"Game {i+1}/{num_games}...", end="", flush=True)
        start_time = time.time()
        winner = play_game(p1, p2)
        duration = time.time() - start_time
        
        if winner == trained_id:
            results[1] += 1
            res_str = "TRAINED WON"
        elif winner == random_id:
            results[2] += 1
            res_str = "RANDOM WON"
        else:
            results["draw"] += 1
            res_str = "DRAW"
            
        print(f" {res_str} ({duration:.1f}s)")
        
    print("-" * 50)
    print("FINAL RESULTS:")
    print(f"Trained AI Wins: {results[1]} ({results[1]/num_games*100:.1f}%)")
    print(f"Random AI Wins:  {results[2]} ({results[2]/num_games*100:.1f}%)")
    if results["draw"] > 0:
        print(f"Draws:          {results['draw']}")
    print("-" * 50)

if __name__ == "__main__":
    import sys
    games = 5
    if len(sys.argv) > 1:
        games = int(sys.argv[1])
    run_benchmark(games)
