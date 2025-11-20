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

train_img_dir = r"D:\CHASE_DB1_SPLIT\train\images"
train_mask_dir = r"D:\CHASE_DB1_SPLIT\train\masks_2nd"

val_img_dir = r"D:\CHASE_DB1_SPLIT\val\images"
val_mask_dir = r"D:\CHASE_DB1_SPLIT\val\masks_2nd"

batch_size = 8
num_epochs = 400
lr = 1e-4
save_path = "isic_teacher_full_db2nd.pth"


def set_seed(s=0):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------- Dataset ----------------
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class ChaseSegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        """
        img_dir:  e.g. D:\\CHASE_DB1_SPLIT\\train\\images
        mask_dir: e.g. D:\\CHASE_DB1_SPLIT\\train\\masks_1st
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

        # list of image filenames (not full paths)
        self.img_names = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ])

        if len(self.img_names) == 0:
            raise RuntimeError(f"No images found in {img_dir}")

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]                # e.g. "Image_01L.jpg"
        img_path = os.path.join(self.img_dir, img_name)

        # derive mask name from image name: Image_01L.jpg -> Image_01L_1stHO.png
        base = os.path.splitext(img_name)[0]          # "Image_01L"
        mask_name = base + "_2ndHO.png"
        mask_path = os.path.join(self.mask_dir, mask_name)

        # safety: check mask exists
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = (np.array(Image.open(mask_path).convert("L")) > 127).astype(np.float32)

        # if you have albumentations transforms
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask



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

    train_ds = ChaseSegDataset(train_img_dir, train_mask_dir, transform=transform)
    val_ds = ChaseSegDataset(val_img_dir,   val_mask_dir,   transform=transform)

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
        if true.dim() == 3:
               true = true.unsqueeze(1)   # [B, 1, H, W]

        true = true.float()

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
