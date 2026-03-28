import dataclasses
from typing import Set, Dict, Tuple, List, Optional
import collections

@dataclasses.dataclass(frozen=True)
class Hex:
    q: int
    r: int

    @property
    def s(self) -> int:
        return -self.q - self.r

    def distance(self, other: 'Hex') -> int:
        return (abs(self.q - other.q) + 
                abs(self.q + self.r - other.q - other.r) + 
                abs(self.r - other.r)) // 2

    def neighbors(self) -> List['Hex']:
        directions = [
            Hex(1, 0), Hex(1, -1), Hex(0, -1),
            Hex(-1, 0), Hex(-1, 1), Hex(0, 1)
        ]
        return [Hex(self.q + d.q, self.r + d.r) for d in directions]

class HeXOEngine:
    def __init__(self):
        self.board: Dict[Hex, int] = {}  # Hex -> player_id (1 or 2)
        self.turn_number = 1  # 1st turn, 2nd turn...
        self.current_player = 1
        self.moves_made_this_turn = 0
        self.game_over = False
        self.winner = None
        
        # Cache of hexes reachable from current board state
        # Rule: Only update between players
        self.cached_reachable_hexes: Set[Hex] = set()

    def get_moves_allowed(self) -> int:
        if self.turn_number == 1:
            return 1
        return 2

    def place_stone(self, hex_coord: Hex) -> bool:
        if self.game_over:
            return False
        
        if hex_coord in self.board:
            return False

        # Apply Radius Rule: Must be within 8 of closest hex (if any exist)
        if self.board:
            # On first turn, if we haven't updated cache, we must check.
            # But we update cache at end of turn.
            if not self.cached_reachable_hexes:
               self._update_reachable_cache()
            
            if hex_coord not in self.cached_reachable_hexes:
                return False

        self.board[hex_coord] = self.current_player
        self.moves_made_this_turn += 1
        
        # Check for win condition
        if self._check_win(hex_coord):
            self.game_over = True
            self.winner = self.current_player
            return True

        # Check if turn is over
        if self.moves_made_this_turn >= self.get_moves_allowed():
            self._end_turn()

        return True

    def _end_turn(self):
        self.current_player = 3 - self.current_player
        self.turn_number += 1
        self.moves_made_this_turn = 0
        self._update_reachable_cache()

    def _update_reachable_cache(self):
        """Updates the set of hexes within 8 spaces of any occupied hex."""
        if not self.board:
            # If no stones, allow starting anywhere? Usually (0,0) is best.
            self.cached_reachable_hexes = {Hex(0, 0)}
            return

        new_cache = set()
        # For simplicity, we could find all hexes with distance <= 8 of any board stone.
        # This could be expensive. Optimization likely needed.
        for stone_hex in self.board:
            # All hexes within distance 8 of stone_hex
            for q in range(-8, 9):
                for r in range(max(-8, -q - 8), min(8, -q + 8) + 1):
                    target = Hex(stone_hex.q + q, stone_hex.r + r)
                    if target not in self.board:
                        new_cache.add(target)
        self.cached_reachable_hexes = new_cache

    def _check_win(self, last_hex: Hex) -> bool:
        directions = [(1, 0), (1, -1), (0, -1)]  # q, r
        player = self.board[last_hex]
        
        for dq, dr in directions:
            count = 1
            # Forward
            curr_q, curr_r = last_hex.q + dq, last_hex.r + dr
            while self.board.get(Hex(curr_q, curr_r)) == player:
                count += 1
                curr_q += dq
                curr_r += dr
            # Backward
            curr_q, curr_r = last_hex.q - dq, last_hex.r - dr
            while self.board.get(Hex(curr_q, curr_r)) == player:
                count += 1
                curr_q -= dq
                curr_r -= dr
            
            if count >= 6:
                return True
        return False

    def get_legal_moves(self) -> List[Hex]:
        if not self.board:
            return [Hex(0, 0)]
        if not self.cached_reachable_hexes:
            self._update_reachable_cache()
            
        # Return all in cache that are empty
        valid_moves = [h for h in self.cached_reachable_hexes if h not in self.board]
        # In the exceedingly rare exact-fill case, fallback
        if not valid_moves:
             return [] 
        return sorted(valid_moves, key=lambda h: (h.q, h.r))
