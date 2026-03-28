import pygame
import math
import sys
import random
import time
from hexo_engine import HeXOEngine, Hex
from best_ai import HeXOBestAI
from train import BOARD_SIZE

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
        self.zoom_scale = 1.0
        
        self.font = pygame.font.SysFont("Arial", 28)
        self.small_font = pygame.font.SysFont("Arial", 20)
        
        self.trained_ai_id = 1
        self.random_ai_id = 2
        
        self.best_ai = HeXOBestAI(1) # Internal model reference
        self.random_ai = RandomAI(2)
        
        self.scores = {1: 0, 2: 0} # Using internal IDs (1 for Trained, 2 for Random)
        self.games_played = 0
        self.reset_game()

    def reset_game(self):
        self.engine = HeXOEngine(boundary_radius=BOARD_SIZE // 2)
        self.games_played += 1
        
        # Alternate who goes first in the engine
        # In engine, player 1 always goes first.
        # We assign our AI objects to those player ids.
        if self.games_played % 2 != 0:
            # Game 1, 3, 5... Trained is P1, Random is P2
            self.p1_is_trained = True
            self.ais = {1: self.best_ai, 2: self.random_ai}
            self.best_ai.player_id = 1
            self.random_ai.player_id = 2
        else:
            # Game 2, 4, 6... Random is P1, Trained is P2
            self.p1_is_trained = False
            self.ais = {1: self.random_ai, 2: self.best_ai}
            self.random_ai.player_id = 1
            self.best_ai.player_id = 2
            
        self.game_over_timer = 0

    def draw_hex(self, h: Hex, color, width=0):
        # Apply zoom to the base radius
        dynamic_radius = HEX_RADIUS * self.zoom_scale
        
        px, py = hex_to_pixel(h, dynamic_radius)
        px += self.camera_x
        py += self.camera_y
        points = []
        for i in range(6):
            angle_deg = 60 * i
            angle_rad = math.pi / 180 * angle_deg
            points.append((px + dynamic_radius * math.cos(angle_rad),
                           py + dynamic_radius * math.sin(angle_rad)))
        pygame.draw.polygon(self.screen, color, points, width)

    def run(self):
        clock = pygame.time.Clock()
        running = True
        
        while running:
            self.screen.fill(BG_COLOR)
            
            # Key Handling (Continuous for smooth panning)
            keys = pygame.key.get_pressed()
            move_speed = 10
            if keys[pygame.K_LEFT]:  self.camera_x += move_speed
            if keys[pygame.K_RIGHT]: self.camera_x -= move_speed
            if keys[pygame.K_UP]:    self.camera_y += move_speed
            if keys[pygame.K_DOWN]:  self.camera_y -= move_speed
            
            # Zoom keys
            if keys[pygame.K_EQUALS] or keys[pygame.K_KP_PLUS]:
                 self.zoom_scale *= 1.05
            if keys[pygame.K_MINUS] or keys[pygame.K_KP_MINUS]:
                 self.zoom_scale *= 0.95

            # Event Handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                if event.type == pygame.MOUSEWHEEL:
                    if event.y > 0: # Scroll Up
                        self.zoom_scale *= 1.1
                    else: # Scroll Down
                        self.zoom_scale *= 0.9
                
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
                    # Was the winner of this game the trained one?
                    winner_is_trained = (self.p1_is_trained and winner == 1) or (not self.p1_is_trained and winner == 2)
                    if winner_is_trained:
                        self.scores[1] += 1
                    else:
                        self.scores[2] += 1
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

            # Overlay Background
            pygame.draw.rect(self.screen, (20, 20, 25, 150), (10, 10, 480, 310))
            
            # Stats calculation
            t_win_rate = (self.scores[1] / self.games_played * 100) if self.games_played > 0 else 0
            
            p1_label = "TRAINED (Neural)" if self.p1_is_trained else "RANDOM (Bot)"
            p2_label = "RANDOM (Bot)" if self.p1_is_trained else "TRAINED (Neural)"
            
            status = f"Game #{self.games_played} | Player {self.engine.current_player}'s Turn"
            if self.engine.game_over:
                winner_is_trained = (self.p1_is_trained and self.engine.winner == 1) or (not self.p1_is_trained and self.engine.winner == 2)
                winner_name = "TRAINED" if winner_is_trained else "RANDOM"
                status = f"GAME OVER - {winner_name} WINS!"

            lines = [
                f"BENCHMARK: {status}",
                f"P1 (Red): {p1_label}",
                f"P2 (Blue): {p2_label}",
                "",
                f"SCOREBOARD (Total Games: {self.games_played})",
                f"Trained Wins: {self.scores[1]}",
                f"Random Wins:  {self.scores[2]}",
                f"Trained Win Rate: {t_win_rate:.1f}%",
                "Controls: Arrows to Pan | Wheel or +/- to Zoom"
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

        pygame.quit()

if __name__ == "__main__":
    gui = HeXOGUIBenchmark()
    gui.run()
