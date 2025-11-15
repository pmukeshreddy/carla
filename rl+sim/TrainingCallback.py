class TrainingCallback(BaseCallback):
    """
    Custom callback for logging and visualization
    """
    def __init__(self, check_freq=1000, save_path='./logs/', verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.best_mean_reward = -np.inf
        
        os.makedirs(save_path, exist_ok=True)
    
    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            # Log training progress
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
                mean_length = np.mean([ep_info["l"] for ep_info in self.model.ep_info_buffer])
                
                print(f"\nStep {self.num_timesteps}:")
                print(f"  Mean Reward: {mean_reward:.2f}")
                print(f"  Mean Length: {mean_length:.2f}")
                
                # Save best model
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    self.model.save(os.path.join(self.save_path, 'best_model'))
                    print(f"  ✓ New best model saved! (reward: {mean_reward:.2f})")
        
        return True

print("✓ Training callback ready")
