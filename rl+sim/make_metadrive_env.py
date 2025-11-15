def make_metadrive_env():
    """
    Create MetaDrive environment with camera enabled
    """
    config = {
        # Environment
        "manual_control": False,
        "use_render": False,  # No 3D window (headless)
        "offscreen_render": True,  # Enable camera rendering
        
        # Camera settings
        "image_observation": True,
        "rgb_clip": True,  # Ensure RGB format
        "image_on_cuda": True,  # Use GPU for rendering (faster on P100!)
        
        # Map settings
        "map": "SSSS",  # S=Straight, O=Circular, C=Curve, r=Ramp
        "num_scenarios": 10,  # Procedural generation
        "start_seed": 0,
        
        # Traffic
        "traffic_density": 0.1,
        "accident_prob": 0.0,
        
        # Physics
        "physics_world_step_size": 0.02,
        "decision_repeat": 5,  # Action repeat (5 steps per action)
        
        # Episode
        "horizon": 1000,  # Max steps per episode
        
        # Vehicle
        "vehicle_config": {
            "image_source": "rgb_camera",
        },
    }
    
    # Create base environment
    base_env = MetaDriveEnv(config=config)
    
    # Wrap with lane segmentation
    env = LaneSegmentationEnv(base_env, lane_predictor, visualize=False)
    
    return env

print("✓ Environment factory ready")
