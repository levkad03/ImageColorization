import os

import torch
from torch.utils.data import DataLoader, random_split
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

                save_image(
                    original[i], os.path.join(output_dir, f"original_{idx}_{i}.jpg")
                )

                save_image(pred_img, os.path.join(output_dir, img_name))

                saved_count += 1

    model.train()


def create_dataloader(
    dataset,
    batch_size=32,
    num_workers=2,
    pin_memory=True,
    train_split=0.8,
    random_seed=123,
):
    torch.manual_seed(random_seed)

    total_size = len(dataset)
    train_size = int(train_split * total_size)
    test_size = total_size - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, test_loader
