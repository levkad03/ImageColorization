import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from autoencoder import AutoEncoder
from dataset import ColorizationDataset
from utils import create_dataloader, load_checkpoint, save_checkpoint, save_predictions

seed = 123

torch.manual_seed(seed)

LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
WEIGHT_DECAY = 0
EPOCHS = 100
NUM_WORKERS = 2
PIN_MEMORY = True
DATA_PATH = "data"
LOAD_MODEL = False
OUTPUT_DIR = "predictions"


def train_fn(loader, model, optimizer, loss_fn, scaler, epoch):
    loop = tqdm(loader, desc=f"Epoch: {epoch}")
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.to(device=DEVICE)

        # forward
        with torch.amp.autocast(DEVICE):
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main():
    model = AutoEncoder().to(DEVICE)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scaler = torch.amp.GradScaler(DEVICE)

    dataset = ColorizationDataset(DATA_PATH)

    train_loader, test_loader = create_dataloader(dataset, BATCH_SIZE)

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    for epoch in range(EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler, epoch)

        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        save_predictions(model, test_loader, OUTPUT_DIR, DEVICE, n_samples=20)


if __name__ == "__main__":
    main()
