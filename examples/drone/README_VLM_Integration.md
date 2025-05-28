# VLM-Guided Drone Simulation

This modified version of the MultiHoverEnv integrates Vision Language Model (VLM) decision-making for intelligent drone waypoint planning and target assignment.

## Key Features

### VLM Integration
- **Real-time Decision Making**: VLM analyzes current drone states and visual scene data every timestep (or at specified intervals)
- **Dynamic Target Assignment**: Drones can be reassigned to different targets based on VLM decisions
- **Collision Avoidance**: VLM considers minimum safe distances between drones when planning waypoints
- **Visual Scene Understanding**: VLM processes camera images to make informed spatial decisions

### Input to VLM
The VLM receives the following data each timestep:
```json
{
  "step": 42,
  "targets": [[x1, y1, z1], [x2, y2, z2], ...],
  "completed": [false, true, false, ...],
  "drone_states": [
    {
      "drone_id": 0,
      "position": [x, y, z],
      "velocity": [vx, vy, vz],
      "attitude": [roll, pitch, yaw]
    },
    ...
  ],
  "min_distance": 0.3,
  "target_threshold": 0.1,
  "n_drones": 3
}
```

### Output from VLM
The VLM returns waypoints and assignments:
```json
{
  "assignments": [
    {"drone_id": 0, "target_id": 1},
    {"drone_id": 1, "target_id": 0},
    {"drone_id": 2, "target_id": 2}
  ],
  "waypoints": [
    {"drone_id": 0, "waypoint": [0.1, 0.5, 0.1]},
    {"drone_id": 1, "waypoint": [0.7, 0.2, 0.2]},
    {"drone_id": 2, "waypoint": [-0.7, 0.2, 1.0]}
  ],
  "n_steps": 20,
  "reasoning": "Explanation of the planning decisions..."
}
```

## Usage

### Basic Usage
```python
from multi_hover_env_with_vlm import MultiHoverEnv

# Define target points
targets = [[1.0, 1.0, 0.5], [-1.0, 1.0, 0.7], [0.0, -1.0, 0.3]]

# Create environment with VLM
env = MultiHoverEnv(
    target_trajectory=targets,
    n_drones=3,
    use_vlm=True,
    vlm_replan_interval=5,  # Call VLM every 5 steps
    min_distance=0.3,       # Safe distance between drones
    target_threshold=0.1,   # Target reach threshold
    picture_save_path="./pictures",
    state_params_save_path="./states",
    rl_checkpoint_path="path/to/rl/checkpoint",
    ckpt=100
)

# Run simulation
env.run()
```

### Configuration Parameters

- `use_vlm`: Enable/disable VLM planning (bool, default: True)
- `vlm_replan_interval`: Steps between VLM calls (int, default: 1)
- `min_distance`: Minimum safe distance between drones (float, default: 0.3)
- `target_threshold`: Distance threshold to consider target reached (float, default: 0.1)

## Requirements

1. **OpenAI API Key**: Set your OpenAI API key as an environment variable
2. **Dependencies**: 
   - openai
   - genesis
   - numpy
   - opencv-python
   - torch
   - scipy

## Files Modified

1. **multi_hover_env_with_vlm.py**: Main environment with VLM integration
   - Added VLM API integration
   - Modified step function for VLM-guided planning
   - Added fallback mechanisms for VLM failures

2. **testLLM.py**: VLM API interface (imported)
   - Handles OpenAI API calls
   - Formats prompts and parses responses

3. **test_vlm_integration.py**: Example usage script

## How it Works

1. **Initialization**: Environment sets up drones, targets, and camera
2. **VLM Planning**: Every `vlm_replan_interval` steps:
   - Capture camera image
   - Collect drone state data
   - Call VLM API with visual and state data
   - Parse VLM response for new waypoints and assignments
3. **Execution**: Drones follow VLM-generated waypoints using RL policy
4. **Fallback**: If VLM fails, uses greedy target assignment
5. **Logging**: Saves all states, VLM responses, and images for analysis

## Benefits of VLM Integration

- **Intelligent Planning**: VLM can understand complex spatial relationships
- **Dynamic Adaptation**: Real-time replanning based on current situations
- **Collision Avoidance**: Considers drone interactions when planning paths
- **Visual Understanding**: Can incorporate visual scene understanding
- **Flexible Assignment**: Can reassign drones to optimize overall performance

## Troubleshooting

- Ensure OpenAI API key is set: `export OPENAI_API_KEY="your-key-here"`
- Check VLM response format matches expected JSON schema
- Monitor console output for VLM call status and fallback usage
- Verify RL checkpoint path and model files exist 