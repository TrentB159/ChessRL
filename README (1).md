# Chess RL Agent

A reinforcement learning agent that learns to play chess using Deep Q-Networks (DQN) and the TorchRL framework. This project builds on concepts and code elements from [CMPT-310-Group-37/Initial-File-Setup](https://github.com/CMPT-310-Group-37/Initial-File-Setup/tree/main), a school project from Simon Fraser University's CMPT 310 (Introduction to Artificial Intelligence) course.

---

## Overview

The agent learns by playing games against itself, receiving rewards for capturing pieces and winning. It uses a convolutional neural network to evaluate board positions and epsilon-greedy exploration to balance learning and exploitation.

---

### Fresh training

```bash
python chessRL_fixed.py --mode train --episodes 10000 --save chess_agent.pth
```

### Resume training from a saved checkpoint

```bash
python chessRL_fixed.py --mode resume --episodes 5000 --save chess_agent.pth --epsilon 0.3
```

Use `--epsilon` to set the starting exploration rate. After initial training, a value of 0.2–0.5 is recommended so the agent exploits what it has learned while still exploring.

### Watch a saved agent play a game

```bash
python chessRL_fixed.py --mode watch --save chess_agent.pth
```

### All options

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `train` | `train`, `resume`, or `watch` |
| `--episodes` | `10000` | Number of training episodes |
| `--save` | `chess_agent.pth` | Path to save or load agent weights |
| `--epsilon` | `None` | Starting epsilon for resumed training |

---

### ChessEnvironment

Raw chess environment wrapping `python-chess`. Handles move validation, board state encoding, and reward computation.

**Board encoding:** 12 channels × 8 × 8 — six piece types (pawn, knight, bishop, rook, queen, king) for each color. Each cell is 1.0 if the piece occupies that square, 0.0 otherwise.

**Reward signal:**
- Piece capture: `material_gain × 0.05` (queen capture ≈ +0.45, pawn capture ≈ +0.05)
- Checkmate win: `+10.0`
- Checkmate loss: `-10.0`
- Draw/stalemate: `-0.5`
- Illegal move: `-10.0` (ends episode)

### ChessNet

Convolutional Q-network:

```
Input: (12, 8, 8)
→ Conv2d(12, 64, 3, padding=1) + BatchNorm + ReLU
→ Conv2d(64, 128, 3, padding=1) + BatchNorm + ReLU
→ Conv2d(128, 128, 3, padding=1) + BatchNorm + ReLU
→ Flatten → Linear(8192, 1024) → ReLU → Dropout(0.3)
→ Linear(1024, 512) → ReLU
→ Linear(512, 218)   ← one Q-value per possible action
```

### Training Loop

- **Algorithm:** DQN with `DQNLoss` from torchrl
- **Replay buffer:** `LazyTensorStorage`, default 50,000 transitions
- **Batch size:** 64
- **Optimizer:** Adam, lr=1e-4
- **Epsilon decay:** 1.0 → 0.05, multiplied by 0.9995 each episode
- **Episode cap:** 150 moves per game to prevent runaway random play

---