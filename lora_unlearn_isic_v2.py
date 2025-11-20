# --------------------------------------------------------------
#  lora_unlearn_isic.py (v2.1 - BG forgetting + guard + weighted forget)
#  LoRA + KD + Background-forget Unlearning for ISIC Segmentation
#  Uses trained teacher from isic_teacher_full.pth
#  TRAIN (on train set only) + EVAL (Retain / Forget / Val)
# --------------------------------------------------------------

import os
import random
import copy
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

import albumentations as A
from albumentations.pytorch import ToTensorV2

import segmentation_models_pytorch as smp
from peft import LoraConfig, get_peft_model

# ---------------- CONFIG ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_img_dir = r"D:\ISIC2018_Task1-2_Training_Input"
train_mask_dir = r"D:\ISIC2018_Task1_Training_GroundTruth"

val_img_dir = r"D:\ISIC2018_Task1-2_Validation_Input"
val_mask_dir = r"D:\ISIC2018_Task1_Validation_GroundTruth"

TEACHER_CKPT = r"isic_teacher_full.pth"  # from train_isic_teacher.py

forget_ratio = 0.10
batch_size   = 64
num_epochs   = 15
lr           = 1e-4
T            = 2.0
alpha        = 1.0           # KD weight on retain
beta_guard   = 0.05          # guard MSE weight on retain (slightly weaker)
lambda_forget = 3.0          # weight for forget loss (stronger forgetting)

LORA_R        = 8
LORA_ALPHA    = 32
LORA_DROPOUT  = 0.05


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
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).float()

        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        elif mask.ndim == 3 and mask.shape[-1] == 1 and mask.shape[0] != 1:
            mask = mask.permute(2, 0, 1)
        mask = mask.contiguous().float()

        return image, mask


transform = A.Compose([
    A.Resize(height=256, width=256),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])


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
    iou  = (intersection + eps) / (pred_sum + mask_sum - intersection + eps)

    return dice.mean().item(), iou.mean().item()


def evaluate_model(model, loader, name, device, thresh=0.5):
    model.eval()
    dices, ious = [], []

    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device, non_blocking=True).float()
            masks = masks.to(device, non_blocking=True).float()
            logits = model(imgs)
            d, i = dice_iou_from_logits(logits, masks, thresh=thresh)
            dices.append(d)
            ious.append(i)

    dice_mean = float(np.mean(dices)) if dices else 0.0
    iou_mean  = float(np.mean(ious)) if ious else 0.0

    print(f"[{name}] Dice={dice_mean:.4f} | IoU={iou_mean:.4f}")
    model.train()
    return dice_mean, iou_mean


def build_unet():
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        classes=1,
        activation=None
    )


# ---------------- MAIN ----------------
if __name__ == "__main__":
    set_seed(0)

    # 1. Train dataset -> split retain/forget
    train_ds = ISICDataset(train_img_dir, train_mask_dir, transform=transform)
    total = len(train_ds)
    forget_sz = int(forget_ratio * total)
    retain_sz = total - forget_sz
    retain_ds, forget_ds = random_split(
        train_ds,
        [retain_sz, forget_sz],
        generator=torch.Generator().manual_seed(0),
    )

    retain_loader = DataLoader(retain_ds, batch_size=batch_size,
                               shuffle=True, num_workers=0, pin_memory=True)
    forget_loader = DataLoader(forget_ds, batch_size=batch_size,
                               shuffle=True, num_workers=0, pin_memory=True)

    retain_eval_loader = DataLoader(retain_ds, batch_size=batch_size,
                                    shuffle=False, num_workers=0, pin_memory=True)
    forget_eval_loader = DataLoader(forget_ds, batch_size=batch_size,
                                    shuffle=False, num_workers=0, pin_memory=True)

    # 2. Separate validation dataset & loader
    val_ds = ISICDataset(val_img_dir, val_mask_dir, transform=transform)
    val_eval_loader = DataLoader(val_ds, batch_size=batch_size,
                                 shuffle=False, num_workers=0, pin_memory=True)

    print(f"Train total: {total} | Retain: {retain_sz} | Forget: {forget_sz}")
    print(f"Val size: {len(val_ds)}")

    # 3. Load teacher
    teacher = build_unet().to(device)
    print(f"Loading TEACHER from: {TEACHER_CKPT}")
    teacher_ckpt = torch.load(TEACHER_CKPT, map_location=device)
    teacher.load_state_dict(teacher_ckpt)
    teacher.eval()

    # 4. Build student as copy + LoRA
    base_student = build_unet().to(device)
    base_student.load_state_dict(teacher_ckpt)

    conv_names = [
        name for name, module in base_student.named_modules()
        if isinstance(module, torch.nn.Conv2d)
    ]

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=conv_names,
        lora_dropout=LORA_DROPOUT,
        bias="none"
    )
    student = get_peft_model(base_student, lora_config).to(device)
    student.print_trainable_parameters()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, student.parameters()),
        lr=lr
    )

    dice_loss = smp.losses.DiceLoss(mode="binary")
    bce_loss  = smp.losses.SoftBCEWithLogitsLoss()

    def seg_loss(pred, true):
        return dice_loss(pred, true) + bce_loss(pred, true)

    # # 5. Baseline teacher metrics
    # print("\n=== Baseline TEACHER Performance (before unlearning) ===")
    evaluate_model(teacher, retain_eval_loader, "Teacher / Retain (Train)", device)
    evaluate_model(teacher, forget_eval_loader, "Teacher / Forget (Train)", device)
    evaluate_model(teacher, val_eval_loader,    "Teacher / Val", device)

    # 6. Unlearning
    print("\n=== Training STUDENT (LoRA) with Retain + BG Forget (weighted) ===")
    student.train()
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        retain_steps = 0

        # ---------- Retain phase: Seg + KD + Guard ----------
        for imgs, masks in retain_loader:
            imgs = imgs.to(device, non_blocking=True).float()
            masks = masks.to(device, non_blocking=True).float()

            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                teacher_logits = teacher(imgs)
            student_logits = student(imgs)

            # Segmentation loss
            loss_seg = seg_loss(student_logits, masks)

            # KD loss
            soft_teacher = torch.sigmoid(teacher_logits / T).detach()
            kd = F.binary_cross_entropy_with_logits(
                student_logits / T, soft_teacher
            ) * (T ** 2)

            # Guard loss: keep student logits close to teacher logits
            guard = F.mse_loss(student_logits, teacher_logits)

            loss = loss_seg + alpha * kd + beta_guard * guard
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            retain_steps += 1

        # ---------- Forget phase: force all-background (weighted) ----------
        for imgs, _ in forget_loader:
            imgs = imgs.to(device, non_blocking=True).float()

            optimizer.zero_grad(set_to_none=True)
            student_logits = student(imgs)

            # All-background target (forget lesion information)
            bg_target = torch.zeros_like(student_logits)
            # Only BCE here; Dice with all-zero mask can be weird
            loss_forget = bce_loss(student_logits, bg_target)

            (lambda_forget * loss_forget).backward()
            optimizer.step()

        avg_loss = epoch_loss / max(retain_steps, 1)
        print(
            f"Epoch {epoch}/{num_epochs} | "
            f"Retain Loss={avg_loss:.4f} | "
            f"(lambda_forget={lambda_forget}, beta_guard={beta_guard})"
        )

    # 7. Metrics after unlearning
    print("\n=== STUDENT Performance After Unlearning ===")
    evaluate_model(student, retain_eval_loader, "Student / Retain (Train)", device)
    evaluate_model(student, forget_eval_loader, "Student / Forget (Train)", device)
    evaluate_model(student, val_eval_loader,    "Student / Val", device)

    # 8. Save LoRA weights
    save_path = "isic_unlearned_lora_cvpr_submission-new-new.pth"
    torch.save(student.state_dict(), save_path)
    print(f"\nUnlearned LoRA model saved to: {save_path}")
