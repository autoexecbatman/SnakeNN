# Snake Neural Network Training

This project implements a Deep Q-Network (DQN) to train an AI agent to play Snake using PyTorch C++ (libtorch) and raylib for visualization.

## Project Structure

- `src/snake_logic.h/cpp` - Core Snake game logic
- `src/snake_game.cpp` - Human-playable Snake game with raylib
- `src/neural_network.h/cpp` - DQN implementation for training
- `src/snake_trainer.cpp` - Training script
- `src/snake_test.cpp` - Test trained AI with visualization

## Dependencies

- **CMake** (3.18+)
- **vcpkg** for package management
- **raylib** for game visualization
- **LibTorch** (CUDA-enabled) for neural network training

## Setup

1. Install raylib via vcpkg:
```bash
vcpkg install raylib:x64-windows
```

2. Ensure LibTorch is available at `D:/libtorch-cuda/libtorch/`

3. Build with CMake:
```bash
mkdir build
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=E:/dev/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build . --config Debug
```

## Usage

### Manual Snake Game
```bash
./Debug/SnakeGame.exe
```
Use arrow keys to control the snake.

### Train AI
```bash
./Debug/SnakeTrainer.exe
```
Trains a DQN for 2000 episodes and saves the model as `snake_model.pt`.

### Test Trained AI
```bash
./Debug/SnakeTest.exe
```
- Manual mode: Use arrow keys
- AI mode: Press SPACE to toggle AI control

## Neural Network Architecture

- **Input**: 14-dimensional state vector
  - Distance to walls (4 values)
  - Food direction (2 values) 
  - Current direction (4 one-hot values)
  - Danger detection (4 values)

- **Network**: 14 → 128 → 128 → 4 (fully connected)
- **Output**: Q-values for 4 actions (UP, DOWN, LEFT, RIGHT)

## Training Details

- **Algorithm**: Deep Q-Learning with experience replay
- **Epsilon-greedy**: ε starts at 1.0, decays to 0.01
- **Replay buffer**: 10,000 experiences
- **Batch size**: 32
- **Learning rate**: 0.001
- **Discount factor**: 0.95

## Game State Representation

The AI receives a 14-dimensional state vector containing:
1. Normalized distances to walls
2. Normalized food direction vector
3. Current movement direction (one-hot)
4. Collision danger in each direction (binary)

This representation gives the AI spatial awareness and helps it make informed decisions about movement direction.
