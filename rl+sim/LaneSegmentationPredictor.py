class LaneSegmentationPredictor:
    def __init__(self,model,device="cuda",img_size=(256,256)):
        self.model = model
        self.model.eval()
        self.device = device
        self.img_size = img_size

        self.transform = transforms.Compose([transforms.Resize(img_size),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], 
                               [0.229, 0.224, 0.225])])

    @torch.no_grad()
    def predict(self,image):
        if isinstance(image,np.ndarray):
            image = Image.fromarray(image.astype(np.unit8))
        img_tensor = self.transforms(image).unsequeeze(0).to(self.device)
        output = self.model(img_tensor)
        pred = torch.argmax(output,dim=1).squeeze().cpu().numpy()
        return pred

    def extract_features(self,mask):
        h,w = mask.shape

        lane_pixels = (mask==1)

        features = []

        lane_ratio = lane_pixels.sum()/(h*w)

        features.append(lane_ratio)

        has_lane = float(lane_pixels.any())
        features.append(has_lane)

        if lane_pixels.any():
            lane_cols = np.where(lane_pixels)[1]
            lane_center_x = lane_cols.mean() / w
        else:
            lane_center_x = 0.5
        features.append(lane_center_x)

        deviation = abs(lane_center_x-0.5)
        features.append(deviation)

        if lane_pixels.any():
            lane_spread = lane_cols.std()/w
        else:
            lane_spread = 0.0
        features.append(lane_spread)

        
