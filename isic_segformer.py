# --------------------------------------------------------------
#  train_isic_teacher.py
#  Train a baseline UNet teacher on ISIC train set
#  Validate on ISIC validation set
#  Saves: isic_teacher_full.pth
# --------------------------------------------------------------

import os
import random
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

import segmentation_models_pytorch as smp

# ---------------- CONFIG ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_img_dir = r"D:\ISIC2018_Task1-2_Training_Input"
train_mask_dir = r"D:\ISIC2018_Task1_Training_GroundTruth"

val_img_dir = r"D:\ISIC2018_Task1-2_Validation_Input"
val_mask_dir = r"D:\ISIC2018_Task1_Validation_GroundTruth"

batch_size = 8
num_epochs = 40
lr = 1e-4
save_path = "isic_teacher_full.pth"


def set_seed(s=0):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------- Dataset ----------------
class ISICDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = [f for f in os.listdir(img_dir) if f.lower().endswith(".jpg")]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(
            self.mask_dir,
            self.images[idx].replace(".jpg", "_segmentation.png")
        )

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = (np.array(Image.open(mask_path).convert("L")) > 127).astype(np.float32)

        if self.transform:
            aug = self.transform(image=image, mask=mask)
            image = aug["image"]
            mask = aug["mask"]
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).float()

        if mask.ndim == 2:
            mask = mask.unsqueeze(0)  # (1,H,W)

        return image, mask.float()


transform = A.Compose(
    [
        A.Resize(256, 256),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ]
)


# ---------------- Metrics ----------------
def dice_iou_from_logits(logits, masks, thresh=0.5, eps=1e-7):
    probs = torch.sigmoid(logits)
    preds = (probs > thresh).float()

    preds = preds.view(preds.size(0), -1)
    masks = masks.view(masks.size(0), -1)

    intersection = (preds * masks).sum(dim=1)
    pred_sum = preds.sum(dim=1)
    mask_sum = masks.sum(dim=1)

    dice = (2 * intersection + eps) / (pred_sum + mask_sum + eps)
    iou = (intersection + eps) / (pred_sum + mask_sum - intersection + eps)

    return dice.mean().item(), iou.mean().item()


# ---------------- MAIN ----------------
if __name__ == "__main__":
    set_seed(0)

    train_ds = ISICDataset(train_img_dir, train_mask_dir, transform=transform)
    val_ds = ISICDataset(val_img_dir, val_mask_dir, transform=transform)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
    )

    print(f"Train size: {len(train_ds)} | Val size: {len(val_ds)}")

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        classes=1,
        activation=None,
    ).to(device)

    dice_loss = smp.losses.DiceLoss(mode="binary")
    bce_loss = smp.losses.SoftBCEWithLogitsLoss()

    def seg_loss(pred, true):
        return dice_loss(pred, true) + bce_loss(pred, true)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_val_dice = -1.0

    for epoch in range(1, num_epochs + 1):
        # -------- train --------
        model.train()
        train_loss = 0.0
        for imgs, masks in train_loader:
            imgs = imgs.to(device).float()
            masks = masks.to(device).float()

            optimizer.zero_grad()
            logits = model(imgs)
            loss = seg_loss(logits, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= max(len(train_loader), 1)

        # -------- val --------
        model.eval()
        val_loss = 0.0
        dices, ious = [], []

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(device).float()
                masks = masks.to(device).float()

                logits = model(imgs)
                loss = seg_loss(logits, masks)
                val_loss += loss.item()

                d, i = dice_iou_from_logits(logits, masks)
                dices.append(d)
                ious.append(i)

        val_loss /= max(len(val_loader), 1)
        val_dice = float(np.mean(dices)) if dices else 0.0
        val_iou = float(np.mean(ious)) if ious else 0.0

        print(
            f"Epoch {epoch}/{num_epochs} | "
            f"Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f} | "
            f"Val Dice={val_dice:.4f} | Val IoU={val_iou:.4f}"
        )

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), save_path)
            print(f"  New best Dice! Saved teacher to: {save_path}")

    print(f"\nTraining done. Best Val Dice={best_val_dice:.4f}")
    print(f"Final teacher checkpoint: {save_path}")


