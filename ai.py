import random
import time
from hexo_engine import HeXOEngine, Hex, List, Dict, Set

class HeXOAI:
    def __init__(self, player_id: int):
        self.player_id = player_id
        self.opponent_id = 3 - player_id

    def evaluate_board(self, engine: HeXOEngine) -> float:
        return self._evaluate_full(engine)

    def _get_line_score(self, c1, c2):
        score = 0
        if c1 > 0 and c2 == 0:
            s = self._score_line(c1)
            score += s if self.player_id == 1 else -s
        elif c2 > 0 and c1 == 0:
            s = self._score_line(c2)
            score += s if self.player_id == 2 else -s
            
        if self.player_id == 1 and c2 >= 5 and c1 == 0: score -= 200000
        elif self.player_id == 1 and c2 >= 4 and c1 == 0: score -= 5000
        
        if self.player_id == 2 and c1 >= 5 and c2 == 0: score -= 200000
        elif self.player_id == 2 and c1 >= 4 and c2 == 0: score -= 5000
        return score

    def _evaluate_full(self, engine: HeXOEngine):
        line_counts = {}
        directions = [(1, 0), (0, 1), (1, -1)]
        for h, p in engine.board.items():
            for dq, dr in directions:
                for i in range(6):
                    sq, sr = h.q - dq * i, h.r - dr * i
                    key = (sq, sr, dq, dr)
                    if key not in line_counts: line_counts[key] = [0, 0]
                    line_counts[key][p-1] += 1
        
        score = 0
        for (c1, c2) in line_counts.values():
            score += self._get_line_score(c1, c2)
        return score

    def _score_line(self, count: int) -> int:
        weights = [0, 1, 10, 100, 1000, 50000, 1000000]
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

    def choose_move(self, engine: HeXOEngine, time_limit: float = 2.5) -> List[Hex]:
        moves_needed = engine.get_moves_allowed() - engine.moves_made_this_turn
        sim_engine = engine.clone()
        moves = []
        for _ in range(moves_needed):
            best_move = self.iterative_deepening_search(sim_engine, time_limit=time_limit)
            if best_move:
                moves.append(best_move)
                sim_engine.place_stone(best_move)
            else:
                break
        return moves

    def _get_initial_line_counts(self, engine: HeXOEngine):
        line_counts = {}
        directions = [(1, 0), (0, 1), (1, -1)]
        for h, p in engine.board.items():
            for dq, dr in directions:
                for i in range(6):
                    sq, sr = h.q - dq * i, h.r - dr * i
                    key = (sq, sr, dq, dr)
                    if key not in line_counts: line_counts[key] = [0, 0]
                    line_counts[key][p-1] += 1
        return line_counts

    def iterative_deepening_search(self, engine: HeXOEngine, time_limit: float) -> Hex:
        start_time = time.time()
        candidates = self._get_pruned_candidates(engine)
        if not candidates: return None

        line_counts = self._get_initial_line_counts(engine)
        current_score = 0
        for (c1, c2) in line_counts.values():
            current_score += self._get_line_score(c1, c2)

        move_scores = []
        for h in candidates:
             delta = self._get_move_delta(h, engine.current_player, line_counts)
             move_scores.append((current_score + delta, h))
        
        random.shuffle(move_scores)
        candidates = [h for _, h in sorted(move_scores, key=lambda x: x[0], reverse=True)]

        best_move = candidates[0]
        depth = 1
        
        while True:
            alpha = -float('inf')
            beta = float('inf')
            results = []
            
            for h in candidates:
                 if time.time() - start_time > time_limit and depth > 1:
                     return best_move

                 delta = self._get_move_delta(h, engine.current_player, line_counts)
                 self._update_line_counts(h, engine.current_player, line_counts)
                 
                 old_state = (engine.moves_made_this_turn, engine.turn_number, engine.current_player, engine.game_over)
                 engine.board[h] = engine.current_player
                 engine.moves_made_this_turn += 1
                 if engine.moves_made_this_turn >= engine.get_moves_allowed():
                     engine.current_player = 3 - engine.current_player
                     engine.turn_number += 1
                     engine.moves_made_this_turn = 0
                 
                 val = self.alpha_beta(engine, depth - 1, alpha, beta, start_time, time_limit, line_counts, current_score + delta)
                 
                 engine.board.pop(h)
                 engine.moves_made_this_turn, engine.turn_number, engine.current_player, engine.game_over = old_state
                 self._update_line_counts(h, engine.current_player, line_counts, undo=True)
                 
                 if val is None: return best_move
                 results.append((val, h))
                 if val > alpha: alpha = val
            
            results.sort(key=lambda x: x[0], reverse=True)
            best_move = results[0][1]
            candidates = [r[1] for r in results]

            if results[0][0] >= 900000: break
            depth += 1
            if depth > 4: break
                
        return best_move

    def _get_move_delta(self, h, player, line_counts):
        delta = 0
        directions = [(1, 0), (0, 1), (1, -1)]
        for dq, dr in directions:
            for i in range(6):
                sq, sr = h.q - dq * i, h.r - dr * i
                key = (sq, sr, dq, dr)
                c1, c2 = line_counts.get(key, [0, 0])
                old_val = self._get_line_score(c1, c2)
                if player == 1: c1 += 1
                else: c2 += 1
                new_val = self._get_line_score(c1, c2)
                delta += (new_val - old_val)
        return delta

    def _update_line_counts(self, h, player, line_counts, undo=False):
        directions = [(1, 0), (0, 1), (1, -1)]
        for dq, dr in directions:
            for i in range(6):
                sq, sr = h.q - dq * i, h.r - dr * i
                key = (sq, sr, dq, dr)
                if key not in line_counts: line_counts[key] = [0, 0]
                if not undo:
                    line_counts[key][player-1] += 1
                else:
                    line_counts[key][player-1] -= 1

    def alpha_beta(self, engine, depth, alpha, beta, start_time, time_limit, line_counts, current_score):
        if time.time() - start_time > time_limit: return None
        if engine.game_over or abs(current_score) > 900000:
             return current_score

        if depth == 0: return current_score

        candidates = self._get_pruned_candidates(engine)
        if not candidates: return current_score

        is_max = (engine.current_player == self.player_id)
        if is_max:
            max_eval = -float('inf')
            for h in candidates:
                delta = self._get_move_delta(h, engine.current_player, line_counts)
                self._update_line_counts(h, engine.current_player, line_counts)
                old_state = (engine.moves_made_this_turn, engine.turn_number, engine.current_player, engine.game_over)
                engine.board[h] = engine.current_player
                engine.moves_made_this_turn += 1
                if engine.moves_made_this_turn >= engine.get_moves_allowed():
                    engine.current_player = 3 - engine.current_player
                    engine.turn_number += 1
                    engine.moves_made_this_turn = 0
                
                eval = self.alpha_beta(engine, depth - 1, alpha, beta, start_time, time_limit, line_counts, current_score + delta)
                
                engine.board.pop(h)
                engine.moves_made_this_turn, engine.turn_number, engine.current_player, engine.game_over = old_state
                self._update_line_counts(h, engine.current_player, line_counts, undo=True)
                
                if eval is None: return None
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha: break
            return max_eval
        else:
            min_eval = float('inf')
            for h in candidates:
                delta = self._get_move_delta(h, engine.current_player, line_counts)
                self._update_line_counts(h, engine.current_player, line_counts)
                old_state = (engine.moves_made_this_turn, engine.turn_number, engine.current_player, engine.game_over)
                engine.board[h] = engine.current_player
                engine.moves_made_this_turn += 1
                if engine.moves_made_this_turn >= engine.get_moves_allowed():
                    engine.current_player = 3 - engine.current_player
                    engine.turn_number += 1
                    engine.moves_made_this_turn = 0

                eval = self.alpha_beta(engine, depth - 1, alpha, beta, start_time, time_limit, line_counts, current_score + delta)
                
                engine.board.pop(h)
                engine.moves_made_this_turn, engine.turn_number, engine.current_player, engine.game_over = old_state
                self._update_line_counts(h, engine.current_player, line_counts, undo=True)
                
                if eval is None: return None
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha: break
            return min_eval
