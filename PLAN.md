# HeXO Implementation Plan

HeXO is a hexagonal grid game where players aim to get 6-in-a-row.

## Rules Recap
- **Turn Order**: P1 moves 1, then P2 moves 2, then players alternate 2 moves each (2, 2, 2, ...).
- **Win Condition**: 6-in-a-row.
- **Radius Rule**: Moves must be within 8 hexes of any stone that was on the board *at the start of the current player's turn*.

## Architecture
1. **Engine**: Handled in `engine.py`. Uses axial/cube coordinates.
2. **AI**: Handled in `ai.py`. Initially a greedy/heuristic-based AI, with potential to house a neural network.
3. **GUI**: Handled in `gui.py` using Pygame for visualization.

## Tasks
- [x] Implement `Hex` class with coordinate math.
- [x] Implement `Board` class for state management.
- [x] Implement move validation (radius rule).
- [x] Implement win condition (check 6 directions).
- [x] Implement simple AI.
- [x] Create Pygame visualizer.
- [x] Create PyTorch Model (HeXONet) architecture
- [x] Setup RTX Ada optimized self-play MCTS Training loop
