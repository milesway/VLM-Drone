#!/usr/bin/env python3
"""
Test script to verify the optimized MultiHoverEnv with different configurations.
Tests various combinations of control modes, LLM settings, and environments.
"""

import os
import numpy as np
from multi_hover_env_with_vlm import MultiHoverEnv

def create_test_directories(base_path):
    """Create test output directories."""
    picture_dir = os.path.join(base_path, "pictures")
    state_dir = os.path.join(base_path, "states") 
    video_dir = os.path.join(base_path, "videos")
    
    os.makedirs(base_path, exist_ok=True)
    os.makedirs(picture_dir, exist_ok=True)
    os.makedirs(state_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)
    
    return picture_dir, state_dir, video_dir

def test_pid_mode_no_llm():
    """Test PID controller mode without LLM (basic control)."""
    print("\n=== Testing PID mode without LLM ===")
    
    base_path = "./test_outputs/pid_no_llm"
    picture_dir, state_dir, video_dir = create_test_directories(base_path)
    
    # Simple target trajectory
    target_trajectory = [
        [1.0, 1.0, 0.5],
        [-1.0, 1.0, 0.7], 
        [0.0, -1.0, 0.3]
    ]
    
    env = MultiHoverEnv(
        picture_save_path=picture_dir,
        video_save_path=video_dir,
        state_params_save_path=state_dir,
        target_trajectory=target_trajectory,
        env_name="plain",
        save_path=base_path,
        ckpt=None,  # This triggers PID mode
        n_drones=3,
        llm_replan_interval=0,  # Disable LLM (0 means no LLM)
        use_vision=False,
        model_name="gpt-4o",
        min_distance=0.2,
        target_threshold=0.15,
        enable_ray_tracing=False,
        show_viewer=False,  # No viewer for test
    )
    env.run()
    
    print(f"✓ Control mode: {env.control_mode}")
    print(f"✓ LLM enabled: {env.use_llm}")
    print(f"✓ Number of PID controllers: {len(env.controllers)}")
    print(f"✓ Environment: {env.env_name}")
    assert env.control_mode == "pid"
    assert env.use_llm == False
    assert len(env.controllers) == 3
    print("PID mode without LLM test passed!")

def test_rl_mode_with_text_llm():
    """Test PID controller mode with text-only LLM."""
    print("\n=== Testing PID mode with text-only LLM ===")
    
    checkpoint_path = "/home/milesway/research/genesis/VLM-Drone/examples/drone/logs/drone_demo"
    base_path = "./test_outputs/pid_text_llm"
    picture_dir, state_dir, video_dir = create_test_directories(base_path)
    
    target_trajectory = [
        [1.5, 0.5, 0.4],
        [-0.5, 1.5, 0.6]
    ]
    
    env = MultiHoverEnv(
        picture_save_path=picture_dir,
        video_save_path=video_dir,
        state_params_save_path=state_dir,
        target_trajectory=target_trajectory,
        env_name="plain",
        save_path=base_path,
        ckpt=None,  # PID mode
        n_drones=2,
        llm_replan_interval=200,  # LLM enabled, replan every 5 steps
        use_vision=False,  # Text-only LLM
        model_name="gpt-4o",
        min_distance=0.2,
        target_threshold=0.15,
        enable_ray_tracing=False,
        show_viewer=False,
    )
    env.run()
    print(f"✓ Control mode: {env.control_mode}")
    print(f"✓ LLM enabled: {env.use_llm}")
    print(f"✓ Vision enabled: {env.use_vision}")
    print(f"✓ LLM replan interval: {env.llm_replan_interval}")
    assert env.control_mode == "pid"
    assert env.use_llm == True
    assert env.use_vision == False
    print("PID mode with text-only LLM test passed!")

def test_rl_mode_with_vision_llm():
    """Test PID controller mode with vision-enabled LLM."""
    print("\n=== Testing PID mode with vision-enabled LLM ===")
    
    checkpoint_path = "/home/milesway/research/genesis/VLM-Drone/examples/drone/logs/drone_demo"
    base_path = "./test_outputs/pid_vision_llm"
    picture_dir, state_dir, video_dir = create_test_directories(base_path)
    
    target_trajectory = [
        [1.0, 1.0, 0.5],
        [-1.0, 1.0, 0.7], 
        [0.0, -1.0, 0.3]
    ]
    
    env = MultiHoverEnv(
        picture_save_path=picture_dir,
        video_save_path=video_dir,
        state_params_save_path=state_dir,
        target_trajectory=target_trajectory,
        env_name="plain",  # Different environment
        save_path=base_path,
        ckpt=checkpoint_path,  # PID mode
        n_drones=3,
        llm_replan_interval=200,  # Frequent replanning for vision
        use_vision=True,  # Vision-enabled LLM
        model_name="gpt-4o",
        min_distance=0.2,
        target_threshold=0.15,
        enable_ray_tracing=False,
        show_viewer=False,
    )
    env.run()
    print(f"✓ Control mode: {env.control_mode}")
    print(f"✓ LLM enabled: {env.use_llm}")
    print(f"✓ Vision enabled: {env.use_vision}")
    print(f"✓ Environment: {env.env_name}")
    print(f"✓ Ray tracing: {env.enable_ray_tracing}")
    assert env.control_mode == "rl"
    assert env.use_llm == True
    assert env.use_vision == True
    assert env.env_name == "plain"
    print("RL mode with vision-enabled LLM test passed!")
    

def test_rl_mode_with_vision_llm_RTX():
    """Test PID controller mode with vision-enabled LLM."""
    print("\n=== Testing PID mode with vision-enabled LLM ===")
    
    checkpoint_path = "/home/milesway/research/genesis/VLM-Drone/examples/drone/logs/drone_demo"
    base_path = "./test_outputs/pid_vision_llm"
    picture_dir, state_dir, video_dir = create_test_directories(base_path)
    
    target_trajectory = [
        [1.0, 1.0, 0.5],
        [-1.0, 1.0, 0.7], 
        [0.0, -1.0, 0.3]
    ]
    
    env = MultiHoverEnv(
        picture_save_path=picture_dir,
        video_save_path=video_dir,
        state_params_save_path=state_dir,
        target_trajectory=target_trajectory,
        env_name="plain",  # Different environment
        save_path=base_path,
        ckpt=checkpoint_path,  # PID mode
        n_drones=3,
        llm_replan_interval=200,  # Frequent replanning for vision
        use_vision=True,  # Vision-enabled LLM
        model_name="gpt-4o",
        min_distance=0.2,
        target_threshold=0.15,
        enable_ray_tracing=True,
        show_viewer=False,
    )
    env.run()
    print(f"✓ Control mode: {env.control_mode}")
    print(f"✓ LLM enabled: {env.use_llm}")
    print(f"✓ Vision enabled: {env.use_vision}")
    print(f"✓ Environment: {env.env_name}")
    print(f"✓ Ray tracing: {env.enable_ray_tracing}")
    assert env.control_mode == "rl"
    assert env.use_llm == True
    assert env.use_vision == True
    assert env.env_name == "plain"
    print("RL mode with vision-enabled LLM test passed!")


def test_different_models():
    """Test different LLM models."""
    print("\n=== Testing different LLM models ===")
    
    models_to_test = ["gpt-4o", "o4-mini", "gemini-2.5-pro"]
    
    for model in models_to_test:
        print(f"Testing model: {model}")
        
        base_path = f"./test_outputs/model_{model}"
        picture_dir, state_dir, video_dir = create_test_directories(base_path)
        
        target_trajectory = [[0.5, 0.5, 0.4]]
        
        env = MultiHoverEnv(
            picture_save_path=picture_dir,
            video_save_path=video_dir,
            state_params_save_path=state_dir,
            target_trajectory=target_trajectory,
            env_name="plain",
            save_path=base_path,
            ckpt=None,
            n_drones=1,
            llm_replan_interval=200,
            use_vision=False,
            model_name=model,
            min_distance=0.2,
            target_threshold=0.15,
            enable_ray_tracing=False,
            show_viewer=False,
        )
        env.run()
        print(f"✓ Model: {env.model_name}")
        assert env.model_name == model
    
    print("Different models test passed!")

def test_various_environments():
    """Test different environment types."""
    print("\n=== Testing different environments ===")
    
    environments = ["plain", "warehouse"]
    
    for env_name in environments:
        print(f"Testing environment: {env_name}")
        
        base_path = f"./test_outputs/env_{env_name}"
        picture_dir, state_dir, video_dir = create_test_directories(base_path)
        
        target_trajectory = [[0.0, 0.0, 0.5]]
        
        env = MultiHoverEnv(
            picture_save_path=picture_dir,
            video_save_path=video_dir,
            state_params_save_path=state_dir,
            target_trajectory=target_trajectory,
            env_name=env_name,
            save_path=base_path,
            ckpt=None,
            n_drones=1,
            llm_replan_interval=0,  # No LLM for simple test
            use_vision=False,
            model_name="gpt-4o",
            min_distance=0.3,
            target_threshold=0.1,
            enable_ray_tracing=False,
            show_viewer=False,
        )
        env.run()
        print(f"✓ Environment: {env.env_name}")
        assert env.env_name == env_name
    
    print("Different environments test passed!")

def run_all_tests():
    """Run all test cases."""
    print("Starting comprehensive test suite for MultiHoverEnv...")
    
    try:
        # test_pid_mode_no_llm()
        # test_rl_mode_with_text_llm()
        # test_rl_mode_with_vision_llm()  
        test_rl_mode_with_vision_llm_RTX()  
        # test_different_models()
        # test_various_environments()
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1) 