# Drone Examples

This directory contains examples of drone simulations using the Genesis framework.

## PID controller (`pid_env.py`, `pid_eval.py`, `pid_train.py`)
The hover environment (`pid_env.py`) is designed to train multiple drones to reach randomly generated target points stably through LLM-generated way-point planning and fine-tuned PID control. 

### Run the simulation (`pid_eval.py`)
The simulation state (state of the environment (target points and their tracked states) and drones' state parameters) at each step is saved in .json files:
- "step": Current simulation step number.
- "targets": List of target point coordinates, one per drone.
- "completed": Boolean list indicating whether each target has been reached.
- "drone_states": List of per-drone states, each containing:
    - "drone_id": ID of the drone.
    - "step": Current step number.
    - "target": Assigned target coordinates.
    - "position": Current drone position [x, y, z].
    - "velocity": Linear velocity vector [vx, vy, vz].
    - "attitude": Drone orientation [roll, pitch, yaw] in degrees.
    - "pid_params": Current PID gains used by the controller (9 groups of [P, I, D]).
    - "rpms": Commanded RPMs for the four propellers.

Run with:
```bash
python pid_eval.py --n_drones 3 # number of drones == number of targets
```

## Available Examples

### 1. Interactive Drone (`interactive_drone.py`)
A real-time interactive drone simulation where you can control the drone using keyboard inputs:
- ↑ (Up Arrow): Move Forward (North)
- ↓ (Down Arrow): Move Backward (South)
- ← (Left Arrow): Move Left (West)
- → (Right Arrow): Move Right (East)
- Space: Increase Thrust (Accelerate)
- Left Shift: Decrease Thrust (Decelerate)
- ESC: Quit

Run with:
```bash
python interactive_drone.py -v -m
```

### 2. Automated Flight (`fly.py`)
A pre-programmed drone flight simulation that follows a predefined trajectory stored in `fly_traj.pkl`.

Run with:
```bash
python fly.py -v -m
```

### 3. Hover Environment (`hover_env.py`, `hover_train.py`, `hover_eval.py`)

The hover environment (`hover_env.py`) is designed to train a drone to maintain a stable hover position by reaching randomly generated target points. The environment includes:

 - Initialization of the scene and entities (plane, drone and target).
 - Reward functions to provide feedback to the agent based on its performance in reaching the target points.
 - **Command resampling to generate new random target points** and environment reset functionalities to ensure continuous training.

**Acknowledgement**: The reward design is inspired by [Champion-level drone racing using deep
reinforcement learning (Nature 2023)](https://www.nature.com/articles/s41586-023-06419-4.pdf)

#### 3.0 Installation

At this stage, we have defined the environments. Now, we use the PPO implementation from `rsl-rl` to train the policy. First, install all Python dependencies via `pip`:

```bash
pip install --upgrade pip
pip install tensorboard rsl-rl-lib==2.2.4
```

#### 3.1 Training

Train the drone hovering policy using the `HoverEnv` environment.

Run with:

```bash
python hover_train.py -e drone-hovering -B 8192 --max_iterations 301
```

Train with visualization:

```bash
python hover_train.py -e drone-hovering -B 8192 --max_iterations 301 -v
```

#### 3.2 Evaluation

Evaluate the trained drone hovering policy.

Run with:

```bash
python hover_eval.py -e drone-hovering --ckpt 300 --record
```

**Note**: If you experience slow performance or encounter other issues
during evaluation, try removing the `--record` option.

For the latest updates, detailed documentation, and additional resources, visit this repository: [GenesisDroneEnv](https://github.com/KafuuChikai/GenesisDroneEnv).

### 4. Quadcopter Controller (`quadcopter_controller.py` & `fly_route.py`)
A PID-based controller to provide stable point-to-point flight without controlling each rotor. Parameters
have been tuned for controlled flight but not optimized. Stored in `fly_route.py`. Run `fly_route.py` to
test.

Run with:

```bash
python fly_route.py
```

## Technical Details

- The drone model used is the Crazyflie 2.X (`urdf/drones/cf2x.urdf`)
- Base hover RPM is approximately 14468
- Movement is achieved by varying individual rotor RPMs to create directional thrust
- The simulation uses realistic physics including gravity and aerodynamics
- Visualization is optimized for macOS using threaded rendering when run with `-m` flag

## Controls Implementation

The interactive drone uses differential RPM control:
- Forward/Backward movement: Adjusts front/back rotor pairs
- Left/Right movement: Adjusts left/right rotor pairs
- All movements maintain a stable hover while creating directional thrust
- RPM changes are automatically clipped to safe ranges (0-25000 RPM)