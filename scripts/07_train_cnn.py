"""Train a simple CNN on CIFAR-10.

This script matches the architecture used in the Lecture 7 notebook.
It can be used to train from scratch and save a state_dict under ./models.

Example
-------
python scripts/07_train_cnn.py --epochs 10 --device cuda
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def build_model() -> nn.Module:
    """LeNet-style CNN that matches the provided pretrained weights."""

    class SimpleCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(3, 8, kernel_size=5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(8, 16, kernel_size=5)

            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            x = torch.flatten(x, 1)
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    return SimpleCNN()


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss_sum = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss_sum += criterion(outputs, labels).item() * inputs.size(0)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = loss_sum / max(total, 1)
    acc = 100.0 * correct / max(total, 1)
    return avg_loss, acc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output", type=str, default="models")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Data
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_dataset = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root="data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=2)

    # Model
    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader, start=1):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 200 == 0:
                print(f"Epoch {epoch + 1}/{args.epochs} | step {i} | loss {running_loss / 200:.4f}")
                running_loss = 0.0

        test_loss, test_acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch + 1}: test_loss={test_loss:.4f}, test_acc={test_acc:.2f}%")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"simple_cnn_{device.type}.pth"
    torch.save(model.state_dict(), out_path)
    print(f"Saved model weights to: {out_path}")


if __name__ == "__main__":
    main()
