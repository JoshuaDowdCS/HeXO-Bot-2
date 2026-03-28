import pygame
import math
import sys
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

def pixel_to_hex(x: float, y: float, size: float) -> Hex:
    q = (2/3 * x) / size
    r = (-1/3 * x + math.sqrt(3)/3 * y) / size
    # Hex rounding
    s = -q - r
    rq, rr, rs = round(q), round(r), round(s)
    dq, dr, ds = abs(rq - q), abs(rr - r), abs(rs - s)
    if dq > dr and dq > ds:
        rq = -rr - rs
    elif dr > ds:
        rr = -rq - rs
    else:
        rs = -rq - rr
    return Hex(int(rq), int(rr))

class HeXOGUI:
    def __init__(self, engine: HeXOEngine):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("HeXO - Get 6 in a Row")
        self.engine = engine
        self.camera_x = SCREEN_WIDTH // 2
        self.camera_y = SCREEN_HEIGHT // 2
        self.font = pygame.font.SysFont("Arial", 24)
        self.ais = {1: HeXOBestAI(1), 2: HeXOBestAI(2)}
        self.ai_active = {1: False, 2: True} # P2 is AI by default

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
            
            # AI Move
            if not self.engine.game_over and self.ai_active.get(self.engine.current_player):
                # Only if the engine is ready for next move
                moves = self.ais[self.engine.current_player].choose_move(self.engine)
                for m in moves:
                    self.engine.place_stone(m)

            # Events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_1:
                        self.ai_active[1] = not self.ai_active[1]
                    if event.key == pygame.K_2:
                        self.ai_active[2] = not self.ai_active[2]
                    if event.key == pygame.K_r:
                        self.engine = HeXOEngine() # Reset

                if event.type == pygame.MOUSEBUTTONDOWN and not self.engine.game_over:
                    if not self.ai_active.get(self.engine.current_player):
                        mx, my = pygame.mouse.get_pos()
                        hx = pixel_to_hex(mx - self.camera_x, my - self.camera_y, HEX_RADIUS)
                        if self.engine.place_stone(hx):
                            print(f"Human placed stone at {hx}")

            # Draw reachable grid (limit drawing for performance)
            legal_moves = self.engine.get_legal_moves()
            for h in legal_moves:
                self.draw_hex(h, (50, 50, 55), 1)

            # Draw board
            for h, p in self.engine.board.items():
                color = P1_COLOR if p == 1 else P2_COLOR
                self.draw_hex(h, color)
                self.draw_hex(h, BORDER_COLOR, 3)
                
            # Draw mouse hover
            if not self.ai_active.get(self.engine.current_player):
                mx, my = pygame.mouse.get_pos()
                hx = pixel_to_hex(mx - self.camera_x, my - self.camera_y, HEX_RADIUS)
                if hx in legal_moves:
                    self.draw_hex(hx, HIGHLIGHT_COLOR, 2)

            # Info text
            p1_bot_type = "Model" if getattr(self.ais[1], 'use_neural', False) else "Random"
            p2_bot_type = "Model" if getattr(self.ais[2], 'use_neural', False) else "Random"
            
            p1_status = f"[AI - {p1_bot_type}]" if self.ai_active[1] else "[Human]"
            p2_status = f"[AI - {p2_bot_type}]" if self.ai_active[2] else "[Human]"
            
            status = f"Player {self.engine.current_player}'s Turn"
            if self.engine.game_over:
                status = f"PLAYER {self.engine.winner} WINS!"
            
            moves_left = self.engine.get_moves_allowed() - self.engine.moves_made_this_turn
            info_lines = [
                f"{status} | Moves Left: {moves_left}",
                f"P1 (Red): {p1_status} [Press 1 to toggle]",
                f"P2 (Blue): {p2_status} [Press 2 to toggle]",
                "Press R to Reset"
            ]
            for i, line in enumerate(info_lines):
                 text = self.font.render(line, True, (255, 255, 255))
                 self.screen.blit(text, (20, 20 + i * 30))

            pygame.display.flip()
            clock.tick(30)

        pygame.quit()

        pygame.quit()

if __name__ == "__main__":
    engine = HeXOEngine()
    gui = HeXOGUI(engine)
    gui.run()
