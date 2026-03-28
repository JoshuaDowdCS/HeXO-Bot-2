# HeXO-Bot-2

## 🛠️ Infrastructure and Development Workflow
**CRITICAL NOTE**: This repository is developed across multiple machines with a strict separation of concerns:
- **Local Machine (Mac)**: Used exclusively for code development, debugging, and small-scale testing. **NO HEAVY TRAINING HAPPENS HERE.**
- **Remote Workstation (Windows/RTX 2000 Ada)**: All neural network training, long-running self-play sessions, and intensive gpu-accelerated benchmarks are executed on this high-performance machine.

**Development Protocol**:
1. Code changes are made and verified locally on the Mac.
2. Changes are **Pushed** to GitHub.
3. The Remote Workstation **Pulls** the latest training script and starts/resumes the training process (`train.py`).
4. Resultant model binaries (`.pth`) are tracked via Git separately or excluded via `.gitignore` to prevent synchronization bloat.
