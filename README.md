# Pygame DoodleJump
Pygame DoodleJump clone with an added Deep Q-Learning agent. The project can run the original visual game loop, train a DQN agent from game-state observations, save checkpoints, and replay a trained checkpoint in inference mode.

## Table of contents
* [General info](#general-info)
* [Requirements](#requirements)
* [Setup](#setup)
* [Training](#training)
* [Inference](#inference)
* [CPU vs GPU](#cpu-vs-gpu)
* [Checkpoints](#checkpoints)
* [Evaluation Episodes](#evaluation-episodes)

![Screenshot](https://github.com/MykleCode/pygame-doodlejump/blob/main/demo.gif)

## General Info
* No images used for graphics
* Well clean and organised code
* Relatively small code
* RL state includes player motion and nearby platform offsets
* DQN uses replay memory, frame stacking, and a target network

## Requirements
* Python 3.10 or 3.11 recommended
* [Pygame](https://www.pygame.org/news)
* [PyTorch](https://pytorch.org/)

## Setup
```bash
pip install -r requirements.txt
```

Run the training loop:

```bash
python main.py
```

## Training
Visual/local training:

```bash
python main.py
```

Headless training, useful on servers or RunPod:

```bash
HEADLESS=1 python main.py
```

Set how many training episodes run before each checkpoint:

```bash
HEADLESS=1 TRAIN_EPISODES_PER_CHECKPOINT=5000 python main.py
```

## Inference
Run a trained checkpoint without learning or exploration:

```bash
python inference.py --checkpoint dqn_checkpoint.pth --episodes 3
```

Headless inference:

```bash
python inference.py --checkpoint dqn_checkpoint.pth --episodes 100 --headless
```

## CPU vs GPU
The agent uses CUDA automatically when PyTorch can access a compatible GPU.

Check CUDA:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Force CPU even when CUDA is available:

```bash
HEADLESS=1 FORCE_CPU=1 python main.py
```

For RTX 50-series GPUs, install a PyTorch build that supports the GPU architecture, for example CUDA 12.8 wheels:

```bash
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

## Checkpoints
Training saves to:

```text
dqn_checkpoint.pth
```

If this file exists, `main.py` loads it automatically on startup and continues training from it.

## Evaluation Episodes
In non-headless mode, the trainer can show visual evaluation episodes after each training block. The default is:

```text
HEADLESS=0 -> 3 eval episodes
HEADLESS=1 -> 0 eval episodes
```

Override it:

```bash
EVAL_EPISODES=5 python main.py
```

Disable visual evaluation locally:

```bash
EVAL_EPISODES=0 python main.py
```
