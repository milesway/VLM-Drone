# VLM-Guided Drone Simulation

This advanced drone simulation environment integrates Vision Language Model (VLM) decision-making for intelligent multi-drone coordination, collision avoidance, and target assignment with comprehensive logging and conversation history support.

## 🚁 Key Features

### 🤖 Advanced VLM Integration
- **Real-time Decision Making**: VLM analyzes current drone states and visual scene data at configurable intervals
- **Conversation History**: Maintains context across planning cycles for improved decision quality
- **Vision & Text Modes**: Supports both vision-enabled VLM and text-only LLM modes
- **Dynamic Target Assignment**: Drones can be reassigned to different targets based on VLM decisions
- **Intelligent Collision Avoidance**: VLM considers minimum safe distances with smart interval calling
- **Comprehensive Logging**: Detailed conversation logs with performance metrics and analysis

### 🚨 Smart Collision Detection
- **Real-time Monitoring**: Continuous collision risk assessment between all drone pairs
- **Adaptive LLM Calling**: Collision-triggered VLM calls every 20 timesteps (configurable)
- **Immediate Response**: First collision detection triggers immediate VLM intervention
- **Cost Optimization**: Reduces API calls while maintaining responsive collision handling

### 🎛️ Multi-Control Mode Support
- **RL Control**: Uses trained reinforcement learning policies for precise drone control
- **PID Control**: Fallback to PID controllers when no RL checkpoint available
- **Automatic Detection**: Seamlessly switches between control modes based on available resources

### 📊 Comprehensive Logging System
- **Real-time Conversation Logs**: Every VLM interaction saved with detailed metadata
- **Performance Metrics**: Response times, success rates, and error analysis
- **Statistical Summaries**: End-of-simulation analysis with insights
- **Structured File Organization**: Organized logs by environment, model, and mode

## 📋 Input to VLM

The VLM receives comprehensive state information:

```json
{
  "step": 42,
  "targets": [[x1, y1, z1], [x2, y2, z2], ...],
  "completed": [false, true, false, ...],
  "drone_states": [
    {
      "drone_id": 0,
      "step": 42,
      "position": [x, y, z],
      "velocity": [vx, vy, vz],
      "attitude": [roll, pitch, yaw]
    }
  ],
  "completed_drones": [
    {
      "drone_id": 0,
      "assigned_target_id": 1,
      "distance_to_target": 0.05,
      "is_completed": true
    }
  ],
  "min_distance": 0.5,
  "target_threshold": 0.1,
  "n_drones": 3,
  "collision_detected": true,
  "collision_pairs": [
    {
      "drone_1": 0,
      "drone_2": 1,
      "distance": 0.45
    }
  ]
}
```

## 📤 Output from VLM

The VLM returns structured planning decisions:

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
  "reasoning": "Collision detected between drones 0 and 1. Separating them by adjusting waypoints while maintaining progress toward targets..."
}
```

## 🚀 Usage

```
python run.py --env_name warehouse --save_path ./outputs/exp01 --ckpt ./checkpoints/rl_model --n_drones 4 --llm_replan_interval 0 --model_name gpt-4o --min_safe_distance 0.3 --target_threshold 0.1 --show_viewer
```
### Basic VLM-Enabled Simulation

```python
from multi_hover_env_with_vlm import MultiHoverEnv

# Define target points
targets = [[1.0, 1.0, 0.5], [-1.0, 1.0, 0.7], [0.0, -1.0, 0.3]]

# Create environment with VLM
env = MultiHoverEnv(
    picture_save_path="./outputs/pictures",
    video_save_path="./outputs/videos", 
    state_params_save_path="./outputs/states",
    target_trajectory=targets,
    env_name="test_sim",
    save_path="./outputs",
    ckpt="./checkpoints/rl_model",  # RL checkpoint (None for PID mode)
    n_drones=3,
    llm_replan_interval=50,      # Regular LLM calls every 50 steps
    use_vision=True,             # Enable vision-based VLM
    model_name="gpt-4o",         # LLM model to use
    min_distance=0.5,            # Safe distance between drones
    target_threshold=0.1,        # Target reach threshold
    enable_ray_tracing=False,    # Visual effects
    show_viewer=True             # Show simulation window
)

# Run simulation
env.run()
```

### Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `llm_replan_interval` | int | 50 | Steps between regular LLM calls (0 disables LLM) |
| `use_vision` | bool | True | Enable vision-based VLM vs text-only LLM |
| `model_name` | str | "gpt-4o" | OpenAI model to use |
| `min_distance` | float | 0.5 | Minimum safe distance between drones |
| `target_threshold` | float | 0.1 | Distance threshold to consider target reached |
| `ckpt` | str/None | None | Path to RL checkpoint (None uses PID control) |
| `collision_llm_interval` | int | 20 | Steps between collision-triggered LLM calls |

### Advanced Usage Examples

```python
# Text-only LLM mode with PID control
env = MultiHoverEnv(
    # ... basic params ...
    llm_replan_interval=100,
    use_vision=False,           # Text-only mode
    model_name="gpt-4o-mini",   # More cost-effective model
    ckpt=None,                  # Use PID control
    pid_params=custom_pid_gains # Custom PID parameters
)

# Disable LLM entirely (pure PID/RL control)
env = MultiHoverEnv(
    # ... basic params ...
    llm_replan_interval=0,      # Disable LLM
    ckpt="./rl_checkpoint"      # Use RL control only
)
```

## 📁 File Structure

```
vlm/
├── multi_hover_env_with_vlm.py    # Main environment class
├── api.py                         # VLM API interface
├── quadcopter_controller.py       # PID controller implementation
├── test_integration.py            # Comprehensive test suite
├── run.py                         # Example usage script
└── outputs/
    ├── pictures/                  # Camera snapshots
    ├── videos/                    # Simulation recordings
    ├── states/                    # Step-by-step state logs
    └── conversation_logs/         # VLM interaction logs
        ├── {env}_{model}_vision_conversation_log.json
        ├── {env}_{model}_text_conversation_log.json
        ├── {env}_{model}_vision_summary.json
        └── {env}_{model}_text_summary.json
```

## 📊 Logging and Analysis

### Conversation Logs
Each VLM interaction is logged with:
- **Timestamps** and step numbers
- **Complete input data** (drone states, collision info)
- **Full VLM responses** (waypoints, reasoning)
- **Performance metrics** (response time, success rate)
- **Error details** for failed calls

### Summary Statistics
End-of-simulation summaries include:
- **Interaction counts** (total, successful, failed)
- **Call trigger analysis** (scheduled vs collision-triggered)
- **Performance metrics** (average response time)
- **Error analysis** with detailed breakdowns

### Example Log Entry
```json
{
  "step": 25,
  "timestamp": 1703123456.789,
  "datetime": "2023-12-20 15:30:56",
  "llm_call_reason": "collision risk detected (interval: 20 steps)",
  "response_time_seconds": 2.34,
  "success": true,
  "collision_detected": true,
  "collision_pairs": [{"drone_1": 0, "drone_2": 1, "distance": 0.45}],
  "response_analysis": {
    "has_reasoning": true,
    "num_waypoints": 3,
    "reasoning_length": 245
  }
}
```

## 🔧 Requirements

### Environment Setup
```bash
# Install dependencies
pip install openai numpy opencv-python torch scipy python-dotenv

# Set up environment variables
export OPENAI_API_KEY="your-openai-api-key"
export GOOGLE_API_KEY="your-google-api-key"  # For Gemini models
```

### Supported Models
- **OpenAI**: `gpt-4o`, `gpt-4o-mini`
- **Google**: `gemini-2.0-flash-exp` (via OpenAI-compatible API)

## 🧪 Testing

Run the comprehensive test suite:

```bash
python test_integration.py
```

Tests include:
- ✅ RL mode with vision-enabled VLM
- ✅ PID mode with text-only LLM  
- ✅ Pure RL control (no LLM)
- ✅ Pure PID control (no LLM)
- ✅ Error handling and fallbacks

## 🎯 Key Behaviors

### Collision Handling
1. **Detection**: Continuous monitoring of drone pair distances
2. **First Response**: Immediate VLM call on first collision detection
3. **Ongoing Management**: Subsequent calls every 20 timesteps during collisions
4. **Smart Recovery**: Automatic return to normal scheduling when collisions resolved

### Control Mode Selection
1. **RL Mode**: When checkpoint provided and successfully loaded
2. **PID Fallback**: When checkpoint fails to load or not provided
3. **Seamless Operation**: Both modes support full VLM integration

### Error Recovery
1. **VLM Failures**: Automatic fallback to greedy target assignment
2. **API Errors**: Logged with full error details for analysis
3. **Invalid Responses**: Comprehensive validation with helpful error messages

## 🐛 Troubleshooting

### Common Issues
- **API Key**: Ensure `OPENAI_API_KEY` environment variable is set
- **JSON Errors**: Check VLM response format in conversation logs
- **Control Failures**: Verify RL checkpoint path or PID parameters
- **Collision Issues**: Adjust `min_distance` and `collision_llm_interval`

### Debug Output
Monitor console for:
```
⚠️  Collision risk detected: Drones 0 and 1 are 0.454m apart (min: 0.5m)
⏳ Step 5: Collision detected but skipping LLM call (next collision call in 15 steps)
📝 Logged conversation step 0 to ./outputs/conversation_logs/...
📊 Conversation summary saved for analysis
```

## 📈 Performance Tips

1. **Cost Optimization**: Use text-only mode for reduced API costs
2. **Interval Tuning**: Adjust `llm_replan_interval` based on scenario complexity
3. **Model Selection**: Use `gpt-4o-mini` for faster, cheaper responses
4. **Collision Sensitivity**: Tune `min_distance` and `collision_llm_interval` for your use case

## 🔬 Research Applications

This environment is ideal for studying:
- **Multi-agent coordination** with AI planning
- **Human-AI collaboration** in robotics
- **Collision avoidance** algorithms
- **Vision-language model** applications in robotics
- **Hybrid control systems** (RL + classical + AI planning) 