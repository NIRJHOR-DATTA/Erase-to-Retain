# --------------------------------------------------------------
#  train_isic_multiclass_teacher.py
#  Teacher classifier for 8-class ISIC folder dataset
#  Folder structure:
#    ...\Train\<class_name>\*.jpg
#    ...\Test\<class_name>\*.jpg
#  Model: ResNet34 (8-way softmax)
#  Saves: isic_teacher_cls8_full.pth
# --------------------------------------------------------------

import os
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torchvision.models as models

# ---------------- CONFIG ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Root containing Train and Test folders
root_dir = r"D:\seg-unlearn\Skin cancer ISIC The International Skin Imaging Collaboration"

train_root = os.path.join(root_dir, "Train")
val_root   = os.path.join(root_dir, "Test")   # we use Test as validation set

batch_size = 16
num_epochs = 20
lr         = 1e-4

out_ckpt   = "isic_teacher_cls8_full.pth"


def set_seed(s=0):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------- Dataset ----------------
class FolderClassificationDataset(Dataset):
    """
    Expects structure:
      root/
        class_a/
          img1.jpg
          ...
        class_b/
          ...
    Creates label indices 0..num_classes-1 based on sorted class folder names.
    """
    IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

        # List class folders
        self.classes = sorted(
            [d for d in os.listdir(root)
             if os.path.isdir(os.path.join(root, d))]
        )
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # Collect (path, label) pairs
        self.samples = []
        for cls in self.classes:
            cls_dir = os.path.join(root, cls)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(self.IMG_EXTS):
                    path = os.path.join(cls_dir, fname)
                    label = self.class_to_idx[cls]
                    self.samples.append((path, label))

        print(f"[Dataset] root={root}")
        print(f"  classes ({len(self.classes)}): {self.classes}")
        print(f"  num samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = np.array(Image.open(path).convert("RGB"))

        if self.transform is not None:
            aug = self.transform(image=image)
            image = aug["image"]  # (C,H,W) tensor
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        label = torch.tensor(label, dtype=torch.long)  # scalar class index
        return image, label, os.path.basename(path)


# Albumentations transforms
train_transform = A.Compose([
    A.Resize(height=256, width=256),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(height=256, width=256),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])


# ---------------- Model ----------------
def build_resnet34_classifier(num_classes, pretrained=True):
    """
    ResNet34 classifier for multi-class classification.
    Output: (B, num_classes)
    """
    model = models.resnet34(
        weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
    )
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


# ---------------- Metrics ----------------
@torch.no_grad()
def evaluate_classifier(model, loader, device, name="Model"):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    for imgs, labels, _ in loader:
        imgs = imgs.to(device).float()
        labels = labels.to(device)

        logits = model(imgs)  # (B, num_classes)
        loss = criterion(logits, labels)

        running_loss += loss.item() * labels.size(0)

        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / max(total, 1)
    acc = correct / max(total, 1)

    print(f"[{name}] Loss={avg_loss:.4f} | Acc={acc:.4f}")
    return acc, avg_loss


# ---------------- TRAIN LOOP ----------------
if __name__ == "__main__":
    set_seed(0)

    # 1. Datasets & loaders
    train_ds = FolderClassificationDataset(train_root, transform=train_transform)
    val_ds   = FolderClassificationDataset(val_root,   transform=val_transform)

    num_classes = len(train_ds.classes)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size,
                              shuffle=False, num_workers=0, pin_memory=True)

    print(f"Train size: {len(train_ds)} | Val size: {len(val_ds)}")
    print(f"Num classes: {num_classes}")

    # 2. Model, optimizer, loss
    model = build_resnet34_classifier(num_classes=num_classes, pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    # 3. Training
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for imgs, labels, _ in train_loader:
            imgs = imgs.to(device).float()
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(imgs)  # (B,num_classes)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)

            preds = torch.argmax(logits, dim=1)
            running_correct += (preds == labels).sum().item()
            running_total += labels.size(0)

        train_loss = running_loss / max(running_total, 1)
        train_acc  = running_correct / max(running_total, 1)

        # Validation
        val_acc, val_loss = evaluate_classifier(model, val_loader, device, name="Val")

        print(
            f"Epoch {epoch}/{num_epochs} | "
            f"Train Loss={train_loss:.4f} | Train Acc={train_acc:.4f} | "
            f"Val Loss={val_loss:.4f} | Val Acc={val_acc:.4f}"
        )

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "state_dict": model.state_dict(),
                "classes": train_ds.classes,
            }, out_ckpt)
            print(f"  -> New best model saved to {out_ckpt} (Val Acc={val_acc:.4f})")

    print(f"\nTraining finished. Best Val Acc={best_val_acc:.4f}")
    print(f"Class mapping (index -> name):")
    for idx, cls in enumerate(train_ds.classes):
        print(f"  {idx}: {cls}")
