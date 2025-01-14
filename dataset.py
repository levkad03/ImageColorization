import os

import torch
from PIL import Image
from skimage.color import rgb2lab
from torch.utils.data import Dataset
from torchvision import transforms


class ColorizationDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = [
            f for f in os.listdir(image_dir) if f.endswith((".png", ".jpg", ".jpeg"))
        ]

        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                ]
            )
        else:
            self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        img = Image.open(img_path).convert("RGB")

        img_tensor = self.transform(img)

        img_lab = rgb2lab(img_tensor.permute(1, 2, 0).numpy())

        L = torch.from_numpy(img_lab[:, :, 0]) / 50 - 1
        ab = torch.from_numpy(img_lab[:, :, 1:] / 128)

        L = L.unsqueeze(0)
        ab = ab.permute(2, 0, 1)

        return {"L": L, "ab": ab, "orig_img": img_tensor}
