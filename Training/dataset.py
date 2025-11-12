import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode


class LaneSegmentationDataset(Dataset):
    def __init__(self, image_paths, label_paths, img_size=(256, 256), augment=False):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.img_size = img_size
        self.agument = augment  # keeping your original name

        # Image: resize -> tensor -> normalize
        self.img_transforms = transforms.Compose([
            transforms.Resize(img_size, interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])

        # Mask: resize with NEAREST to preserve labels
        self.mask_transforms = transforms.Compose([
            transforms.Resize(img_size, interpolation=InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.label_paths[idx])

        # ---------- AUGMENTATION (on PIL images) ----------
        if self.agument:
            # Horizontal flip
            if random.random() < 0.5:
                image = F.hflip(image)
                mask = F.hflip(mask)

            # Vertical flip
            if random.random() < 0.5:
                image = F.vflip(image)
                mask = F.vflip(mask)

            # Small random rotation
            if random.random() < 0.5:
                angle = random.uniform(-15, 15)
                image = F.rotate(
                    image, angle,
                    interpolation=InterpolationMode.BILINEAR
                )
                mask = F.rotate(
                    mask, angle,
                    interpolation=InterpolationMode.NEAREST
                )

            # Random scale + translation (mild)
            if random.random() < 0.5:
                w, h = image.size
                scale = random.uniform(0.9, 1.1)
                max_trans = 0.05
                tx = int(random.uniform(-max_trans, max_trans) * w)
                ty = int(random.uniform(-max_trans, max_trans) * h)

                image = F.affine(
                    image,
                    angle=0,
                    translate=(tx, ty),
                    scale=scale,
                    shear=0.0,
                    interpolation=InterpolationMode.BILINEAR
                )
                mask = F.affine(
                    mask,
                    angle=0,
                    translate=(tx, ty),
                    scale=scale,
                    shear=0.0,
                    interpolation=InterpolationMode.NEAREST
                )

            # Color jitter (image only)
            if random.random() < 0.5:
                jitter = transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1
                )
                image = jitter(image)

        # ---------- TO TENSOR & NORMALIZE ----------
        image = self.img_transforms(image)
        mask = self.mask_transforms(mask)      # shape: (1, H, W) in [0,1] from 0–255

        # Convert mask to int labels [0..255] -> long
        mask = (mask.squeeze(0) * 255).long()

        return image, mask
