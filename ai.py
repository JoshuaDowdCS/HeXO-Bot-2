import random
import time
from hexo_engine import HeXOEngine, Hex, List, Dict, Set

class HeXOAI:
    def __init__(self, player_id: int):
        self.player_id = player_id
        self.opponent_id = 3 - player_id

    def evaluate_board(self, engine: HeXOEngine) -> float:
        """Simple heuristic score for player_id. Positive is good for self, Negative for opponent."""
        if engine.game_over:
            if engine.winner == self.player_id: return 1000000
            elif engine.winner == self.opponent_id: return -1000000
            return 0

        score = 0
        active_lines = set()
        directions = [(1, 0), (0, 1), (1, -1)]

        for h in engine.board:
            for dq, dr in directions:
                for i in range(6):
                    sq = h.q - dq * i
                    sr = h.r - dr * i
                    active_lines.add((sq, sr, dq, dr))

        for sq, sr, dq, dr in active_lines:
            counts = {1: 0, 2: 0, None: 0}
            for i in range(6):
                p = engine.board.get(Hex(sq + dq * i, sr + dr * i))
                counts[p] += 1
            
            my_c = counts[self.player_id]
            opp_c = counts[self.opponent_id]
            
            if my_c > 0 and opp_c == 0:
                # Scoring for our own potential lines
                score += self._score_line(my_c)
            elif opp_c > 0 and my_c == 0:
                # Penalty for opponent's potential lines
                val = self._score_line(opp_c)
                score -= val
                # Critical defensive panic for threats near win
                if opp_c >= 5:
                    score -= 200000 # Prioritize blocking 5-in-a-row extremely highly
                elif opp_c >= 4:
                    score -= 5000 


        return score

    def _score_line(self, count: int) -> int:
        weights = [0, 1, 10, 100, 1000, 10000, 1000000]
        return weights[min(count, 6)]

    def _get_pruned_candidates(self, engine: HeXOEngine) -> List[Hex]:
        candidates = engine.get_legal_moves()
        if engine.board:
            occupied = set(engine.board.keys())
            interesting = set()
            for h in occupied:
                for q in range(-2, 3):
                    for r in range(max(-2, -q - 2), min(2, -q + 2) + 1):
                        target = Hex(h.q + q, h.r + r)
                        if target not in occupied:
                            interesting.add(target)
            legal_set = set(candidates)
            pruned = list(interesting.intersection(legal_set))
            if pruned:
                return pruned
        return candidates

    def _save_state(self, engine: HeXOEngine):
        return (
            engine.board.copy(), 
            engine.moves_made_this_turn, 
            engine.turn_number, 
            engine.current_player, 
            engine.cached_reachable_hexes.copy() if engine.cached_reachable_hexes else set(),
            engine.game_over,
            engine.winner
        )
        
    def _restore_state(self, engine: HeXOEngine, b, m, t, c, cache, go, w):
        engine.board = b
        engine.moves_made_this_turn = m
        engine.turn_number = t
        engine.current_player = c
        engine.cached_reachable_hexes = cache
        engine.game_over = go
        engine.winner = w

    def choose_move(self, engine: HeXOEngine) -> List[Hex]:
        import copy
        moves_needed = engine.get_moves_allowed() - engine.moves_made_this_turn
        sim_engine = copy.deepcopy(engine)
        moves = []
        for _ in range(moves_needed):
            # Time limit 2.5s per move, meaning 5s per turn max
            best_move = self.iterative_deepening_search(sim_engine, time_limit=2.5)
            if best_move:
                moves.append(best_move)
                sim_engine.place_stone(best_move)
            else:
                break
        return moves

    def iterative_deepening_search(self, engine: HeXOEngine, time_limit: float) -> Hex:
        start_time = time.time()
        
        candidates = self._get_pruned_candidates(engine)
        if not candidates:
            return None

        # Sort candidates quickly
        move_scores = []
        for h in candidates:
             engine.board[h] = self.player_id
             score = self.evaluate_board(engine)
             del engine.board[h]
             move_scores.append((score, h))
        random.shuffle(move_scores)
        candidates = [h for _, h in sorted(move_scores, key=lambda x: x[0], reverse=True)]

        best_move = candidates[0]
        depth = 1
        
        while True:
            alpha = -float('inf')
            beta = float('inf')
            current_best_move = candidates[0]
            current_best_val = -float('inf')
            
            for h in candidates:
                 if time.time() - start_time > time_limit and depth > 1:
                     return best_move

                 old_b, old_m, old_t, old_c, old_cache, old_go, old_w = self._save_state(engine)
                 engine.place_stone(h)
                 val = self.alpha_beta(engine, depth - 1, alpha, beta, start_time, time_limit)
                 self._restore_state(engine, old_b, old_m, old_t, old_c, old_cache, old_go, old_w)
                 
                 if val is None: 
                      return best_move

                 if val > current_best_val:
                     current_best_val = val
                     current_best_move = h
                 alpha = max(alpha, val)

            best_move = current_best_move
            
            candidates.remove(best_move)
            candidates.insert(0, best_move)

            if current_best_val >= 900000:
                break
                
            depth += 1
            if depth > 4: 
                break
                
        return best_move

    def alpha_beta(self, engine: HeXOEngine, depth: int, alpha: float, beta: float, start_time: float, time_limit: float):
        if time.time() - start_time > time_limit:
            return None 

        if engine.game_over:
            if engine.winner == self.player_id:
                return 1000000 + depth * 1000
            elif engine.winner == self.opponent_id:
                return -1000000 - depth * 1000
            return 0
            
        if depth == 0:
            return self.evaluate_board(engine)

        candidates = self._get_pruned_candidates(engine)
        if not candidates:
            return self.evaluate_board(engine)

        is_max = (engine.current_player == self.player_id)

        if is_max:
            max_eval = -float('inf')
            for h in candidates:
                old_state = self._save_state(engine)
                engine.place_stone(h)
                eval = self.alpha_beta(engine, depth - 1, alpha, beta, start_time, time_limit)
                self._restore_state(engine, *old_state)
                
                if eval is None: return None
                
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha: break
            return max_eval
        else:
            min_eval = float('inf')
            for h in candidates:
                old_state = self._save_state(engine)
                engine.place_stone(h)
                eval = self.alpha_beta(engine, depth - 1, alpha, beta, start_time, time_limit)
                self._restore_state(engine, *old_state)
                
                if eval is None: return None
                
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha: break
            return min_eval

