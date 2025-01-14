import os

import numpy as np
import torch
from skimage.color import lab2rgb
from torchvision.utils import save_image


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def save_predictions(model, loader, output_dir, device, n_samples=10):
    os.makedirs(output_dir, exist_ok=True)

    saved_count = 0

    model.eval()
    with torch.no_grad():
        for idx, (gray, original) in enumerate(loader):
            if saved_count >= n_samples:
                break

            gray = gray.to(device=device)
            predictions = model(gray)

            for i in range(gray.shape[0]):
                if saved_count >= n_samples:
                    break

                pred_img = predictions[i].cpu().numpy()
                img_name = f"prediction_{idx}_{i}.jpg"

                pred_img = torch.from_numpy(pred_img).float().cpu()

                save_image(pred_img, os.path.join(output_dir, img_name))

                saved_count += 1

    model.train()
