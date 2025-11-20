# --------------------------------------------------------------
#  lora_unlearn_isic_multiclass.py (Student1 - strong forgetting)
#  LoRA + KD + Entropy-based Forgetting for multi-class ISIC classifier
#  Uses trained teacher from isic_teacher_cls8_full.pth
#  TRAIN (on Train set only) + EVAL (Retain / Forget / Test)
# --------------------------------------------------------------

import os
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torchvision.models as models
from peft import LoraConfig, get_peft_model

# ---------------- CONFIG ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Root containing Train and Test folders
root_dir   = r"D:\seg-unlearn\Skin cancer ISIC The International Skin Imaging Collaboration"
train_root = os.path.join(root_dir, "Train")
test_root  = os.path.join(root_dir, "Test")  # used as validation/test split

TEACHER_CKPT = r"isic_teacher_cls8_full.pth"  # from train_isic_multiclass_teacher.py

forget_ratio  = 0.10
batch_size    = 16
num_epochs    = 30
lr            = 1e-4

T             = 2.0        # KD temperature
alpha         = 1.0        # KD weight on retain
beta_guard    = 0.05       # guard MSE weight on retain (slightly weaker)
lambda_forget = 3.0        # weight for forget loss (stronger forgetting)

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

test_transform = A.Compose([
    A.Resize(height=256, width=256),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])


# ---------------- Model ----------------
def build_resnet34_classifier(num_classes, pretrained=False):
    """
    ResNet34 classifier for multi-class classification.
    Output: (B, num_classes)
    """
    if pretrained:
        weights = models.ResNet34_Weights.IMAGENET1K_V1
    else:
        weights = None
    model = models.resnet34(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def attach_lora(base_model: nn.Module) -> nn.Module:
    conv_names = [
        name for name, module in base_model.named_modules()
        if isinstance(module, nn.Conv2d)
    ]
    cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=conv_names,
        bias="none"
    )
    return get_peft_model(base_model, cfg)


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

        logits = model(imgs)            # (B, num_classes)
        loss = criterion(logits, labels)

        running_loss += loss.item() * labels.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / max(total, 1)
    acc = correct / max(total, 1)

    print(f"[{name}] Loss={avg_loss:.4f} | Acc={acc:.4f}")
    return acc, avg_loss


# ---------------- MAIN ----------------
if __name__ == "__main__":
    set_seed(0)

    # 1. Build Train dataset and deterministic retain/forget split
    full_train_ds = FolderClassificationDataset(train_root, transform=train_transform)
    num_classes = len(full_train_ds.classes)

    total = len(full_train_ds)
    forget_sz = int(forget_ratio * total)
    retain_sz = total - forget_sz

    retain_ds, forget_ds = random_split(
        full_train_ds,
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

    # Test dataset (used as validation)
    test_ds = FolderClassificationDataset(test_root, transform=test_transform)
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                             shuffle=False, num_workers=0, pin_memory=True)

    print(f"Train total: {total} | Retain: {retain_sz} | Forget: {forget_sz}")
    print(f"Test size: {len(test_ds)} | Num classes: {num_classes}")

    # 2. Load TEACHER
    teacher = build_resnet34_classifier(num_classes=num_classes, pretrained=False).to(device)
    print(f"Loading TEACHER from: {TEACHER_CKPT}")
    ckpt = torch.load(TEACHER_CKPT, map_location=device)

    # teacher checkpoint saved as { "state_dict": ..., "classes": [...] }
    teacher.load_state_dict(ckpt["state_dict"])
    teacher_classes = ckpt.get("classes", None)
    if teacher_classes is not None:
        print(f"Teacher classes: {teacher_classes}")
    teacher.eval()

    # 3. Build STUDENT as teacher copy + LoRA (Student1: strong forgetting)
    base_student = build_resnet34_classifier(num_classes=num_classes, pretrained=False).to(device)
    base_student.load_state_dict(ckpt["state_dict"])

    student = attach_lora(base_student).to(device)
    student.print_trainable_parameters()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, student.parameters()),
        lr=lr,
        weight_decay=1e-4
    )

    ce_loss = nn.CrossEntropyLoss()

    # 4. Baseline teacher metrics
    print("\n=== Baseline TEACHER Performance (before unlearning) ===")
    evaluate_classifier(teacher, retain_eval_loader, device, name="Teacher / Retain (Train)")
    evaluate_classifier(teacher, forget_eval_loader, device, name="Teacher / Forget (Train)")
    evaluate_classifier(teacher, test_loader,        device, name="Teacher / Test")

    # 5. Unlearning
    print("\n=== Training STUDENT1 (LoRA) with Retain + Entropy-based Forget (weighted) ===")
    student.train()
    for epoch in range(1, num_epochs + 1):
        running_loss = 0.0
        retain_steps = 0

        # ---------- Retain phase: CE + KD + Guard ----------
        for imgs, labels, _ in retain_loader:
            imgs = imgs.to(device).float()
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                t_logits = teacher(imgs)   # (B, C)

            s_logits = student(imgs)      # (B, C)

            # Classification loss (CrossEntropy)
            loss_cls = ce_loss(s_logits, labels)

            # KD loss: KL divergence between teacher and student softmax at temperature T
            t_probs = F.softmax(t_logits / T, dim=1).detach()
            log_s  = F.log_softmax(s_logits / T, dim=1)
            kd = F.kl_div(log_s, t_probs, reduction="batchmean") * (T ** 2)

            # Guard loss: keep logits close to teacher logits
            guard = F.mse_loss(s_logits, t_logits)

            loss = loss_cls + alpha * kd + beta_guard * guard
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            retain_steps += 1

        # ---------- Forget phase: push predictions toward uniform (max entropy) ----------
        for imgs, labels, _ in forget_loader:
            imgs = imgs.to(device).float()

            optimizer.zero_grad(set_to_none=True)

            s_logits = student(imgs)          # (B, C)
            log_p    = F.log_softmax(s_logits, dim=1)   # log p(y|x)

            # uniform target over classes
            uniform = torch.full_like(log_p, 1.0 / num_classes)

            # KL( uniform || p ) = sum u * (log u - log p )
            # Using PyTorch's KLDivLoss with log_p as input and uniform as target:
            loss_forget = F.kl_div(log_p, uniform, reduction="batchmean")

            (lambda_forget * loss_forget).backward()
            optimizer.step()

        avg_retain_loss = running_loss / max(retain_steps, 1)
        print(
            f"Epoch {epoch}/{num_epochs} | "
            f"Retain Loss={avg_retain_loss:.4f} | "
            f"(lambda_forget={lambda_forget}, beta_guard={beta_guard}, alpha={alpha})"
        )

    # 6. Metrics after unlearning
    print("\n=== STUDENT1 Performance After Unlearning ===")
    evaluate_classifier(student, retain_eval_loader, device, name="Student1 / Retain (Train)")
    evaluate_classifier(student, forget_eval_loader, device, name="Student1 / Forget (Train)")
    evaluate_classifier(student, test_loader,        device, name="Student1 / Test")

    # 7. Save LoRA student weights
    save_path = "isic_unlearned_cls8_lora_student1.pth"
    torch.save(student.state_dict(), save_path)
    print(f"\nUnlearned multi-class LoRA model saved to: {save_path}")
