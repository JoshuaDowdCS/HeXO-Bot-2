import pygame
import math
import sys
import random
import time
from hexo_engine import HeXOEngine, Hex
from best_ai import HeXOBestAI

# UI Constants
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
HEX_RADIUS = 25
BG_COLOR = (30, 30, 35)
HEX_COLOR = (60, 60, 65)
P1_COLOR = (255, 100, 100)
P2_COLOR = (100, 200, 255)
HIGHLIGHT_COLOR = (200, 200, 200)
BORDER_COLOR = (20, 20, 25)

def hex_to_pixel(h: Hex, size: float) -> tuple[float, float]:
    x = size * (3/2 * h.q)
    y = size * (math.sqrt(3)/2 * h.q + math.sqrt(3) * h.r)
    return x, y

class RandomAI:
    def __init__(self, player_id):
        self.player_id = player_id
    def choose_move(self, engine: HeXOEngine):
        moves_needed = engine.get_moves_allowed() - engine.moves_made_this_turn
        moves = []
        sim_engine = engine.clone()
        for _ in range(moves_needed):
            legal = sim_engine.get_legal_moves()
            if not legal: break
            move = random.choice(legal)
            moves.append(move)
            sim_engine.place_stone(move)
        return moves

class HeXOGUIBenchmark:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("HeXO - Trained vs Random Benchmark")
        self.camera_x = SCREEN_WIDTH // 2
        self.camera_y = SCREEN_HEIGHT // 2
        self.font = pygame.font.SysFont("Arial", 28)
        self.small_font = pygame.font.SysFont("Arial", 20)
        
        self.trained_ai_id = 1
        self.random_ai_id = 2
        
        self.ais = {
            self.trained_ai_id: HeXOBestAI(self.trained_ai_id),
            self.random_ai_id: RandomAI(self.random_ai_id)
        }
        
        self.scores = {self.trained_ai_id: 0, self.random_ai_id: 0}
        self.games_played = 0
        self.reset_game()

    def reset_game(self):
        self.engine = HeXOEngine()
        self.games_played += 1
        # Alternate who goes first
        if self.games_played % 2 == 0:
            self.trained_ai_id = 2
            self.random_ai_id = 1
        else:
            self.trained_ai_id = 1
            self.random_ai_id = 2
            
        self.ais[self.trained_ai_id].player_id = self.trained_ai_id
        self.ais[self.random_ai_id].player_id = self.random_ai_id
        self.game_over_timer = 0

    def draw_hex(self, h: Hex, color, width=0):
        px, py = hex_to_pixel(h, HEX_RADIUS)
        px += self.camera_x
        py += self.camera_y
        points = []
        for i in range(6):
            angle_deg = 60 * i
            angle_rad = math.pi / 180 * angle_deg
            points.append((px + HEX_RADIUS * math.cos(angle_rad),
                           py + HEX_RADIUS * math.sin(angle_rad)))
        pygame.draw.polygon(self.screen, color, points, width)

    def run(self):
        clock = pygame.time.Clock()
        running = True
        
        while running:
            self.screen.fill(BG_COLOR)
            
            # Events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        self.reset_game()
                        self.scores = {1: 0, 2: 0}
                        self.games_played = 0

            # AI Move
            if not self.engine.game_over:
                current_player = self.engine.current_player
                moves = self.ais[current_player].choose_move(self.engine)
                for m in moves:
                    self.engine.place_stone(m)
            else:
                # Game is over, wait a bit then reset
                if self.game_over_timer == 0:
                    winner = self.engine.winner
                    # Identify if the winner was trained or random
                    if winner == self.trained_ai_id:
                        self.scores[self.trained_ai_id] += 1
                    else:
                        self.scores[self.random_ai_id] += 1
                    self.game_over_timer = time.time()
                
                if time.time() - self.game_over_timer > 2: # 2 second pause
                    self.reset_game()

            # Draw reachable grid
            legal_moves = self.engine.get_legal_moves()
            for h in legal_moves:
                self.draw_hex(h, (50, 50, 55), 1)

            # Draw board
            for h, p in self.engine.board.items():
                color = P1_COLOR if p == 1 else P2_COLOR
                self.draw_hex(h, color)
                self.draw_hex(h, BORDER_COLOR, 3)

            # Stats overlay
            t_win_rate = (self.scores[self.trained_ai_id] / self.games_played * 100) if self.games_played > 0 else 0
            
            p1_label = "TRAINED (Neural)" if self.trained_ai_id == 1 else "RANDOM (Bot)"
            p2_label = "TRAINED (Neural)" if self.trained_ai_id == 2 else "RANDOM (Bot)"
            
            status = f"Game #{self.games_played} | Player {self.engine.current_player}'s Turn"
            if self.engine.game_over:
                winner_name = "TRAINED" if self.engine.winner == self.trained_ai_id else "RANDOM"
                status = f"GAME OVER - {winner_name} WINS!"

            lines = [
                f"BENCHMARK: {status}",
                f"P1 (Red): {p1_label}",
                f"P2 (Blue): {p2_label}",
                "",
                f"SCOREBOARD (Total Games: {self.games_played})",
                f"Trained Wins: {self.scores[self.trained_ai_id]}",
                f"Random Wins:  {self.scores[self.random_ai_id]}",
                f"Trained Win Rate: {t_win_rate:.1f}%"
            ]
            
            for i, line in enumerate(lines):
                color = (255, 255, 255)
                if "TRAINED" in line: color = (255, 150, 150)
                if "RANDOM" in line: color = (150, 200, 255)
                text = self.font.render(line, True, color)
                self.screen.blit(text, (30, 30 + i * 35))

            pygame.display.flip()
            clock.tick(60)

        pygame.quit()

if __name__ == "__main__":
    gui = HeXOGUIBenchmark()
    gui.run()
