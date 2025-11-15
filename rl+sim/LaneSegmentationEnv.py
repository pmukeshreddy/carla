class LaneSegmentationEnv(gym.Wrapper):
    def __init__(self,env,lane_predictor,visualize=False):
        super().__intit__(env)
        self.lane_predictor = lane_predictor
        self.visualize = visualize

        self.observation_space = spaces.Box(low=0.0,high=1.0,shape=(10,),dtype=np.float32)

        self.last_image = None
        self.last_mask = None

     def _get_lane_features(self,obs):
         camera_image = obs["image"]

         mask = self.lane_predictor.predict(camera_image)

        features = self.lane_predictor.extract_features(mask)

        self.last_image = camera_image
        self.last_mask = mask

        return features

    def reset(self,**kwargs):
        obs,info = self.env.reset(**kwargs)
        features = self._get_lane_features(obs)
        return features,info

    def step(self,action):
        obs,reward,terminated,truncated , info = self.env.step(action)
        features = self._get_lane_features(obs)
        if features[1] == 0:
            reward -= 1

        deviation =  features[3]
        reward -= deviation * 2.0
        return features,reward,terminated ,truncated , info

    def visualize_state(self):
        """Show current image, mask, and overlay"""
        if self.last_image is None:
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(self.last_image)
        axes[0].set_title('Camera View')
        axes[0].axis('off')
        
        # Segmentation mask
        axes[1].imshow(self.last_mask, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title('Lane Segmentation (White=Lane)')
        axes[1].axis('off')
        
        # Overlay
        overlay = self.last_image.copy()
        overlay[self.last_mask == 1] = [0, 255, 0]  # Lane = Green
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay (Green=Lane)')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()

lane_predictor = LaneSegmentationPredictor(lane_model, device=device)
print("✓ Lane predictor ready")
print(f"  Feature dimension: 10")
