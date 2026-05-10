import os
import time
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

# ── Force CPU & Optimize for AMD 4600H ─────────────────────────────────────────
device = torch.device("cpu")
torch.set_num_threads(6)  # AMD 4600H has 6 physical cores; aligns optimal threading
torch.backends.mkldnn.enabled = True # Ensure oneDNN (MKL-DNN) is enabled

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_DIR          = "./oxford-iiit-pet/images"
EPOCHS            = 20
BATCH_SIZE        = 8
IMG_SIZE          = 224
LR                = 0.01
SAMPLES_PER_CLASS = 20
SEED              = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Build class list from image filenames ──────────────────────────────────────
# Gracefully handle missing directory for initial testing
os.makedirs(DATA_DIR, exist_ok=True) 
all_images = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".jpg")]

class_to_files = {}
for fname in all_images:
    parts = fname.rsplit("_", 1)
    if len(parts) == 2 and parts[1].split(".")[0].isdigit():
        cls = parts[0]
        class_to_files.setdefault(cls, []).append(fname)

classes      = sorted(class_to_files.keys())
class_to_idx = {c: i for i, c in enumerate(classes)}
print(f"Found {len(classes)} classes: {classes[:5]} ...")

# ── Sample 20 images per class, then split 80/20 train/val ────────────────────
train_samples, val_samples = [], []
for cls, files in class_to_files.items():
    sampled = random.sample(files, min(SAMPLES_PER_CLASS, len(files)))
    idx     = class_to_idx[cls]
    labeled = [(os.path.join(DATA_DIR, f), idx) for f in sampled]
    split   = max(1, int(0.8 * len(labeled)))
    train_samples += labeled[:split]
    val_samples   += labeled[split:]

print(f"Train: {len(train_samples)} | Val: {len(val_samples)}")

# ── Transforms ─────────────────────────────────────────────────────────────────
train_transform = T.Compose([
    T.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    T.RandomCrop(IMG_SIZE),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
])
val_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
])

# ── Dataset ────────────────────────────────────────────────────────────────────
class PetDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, label

train_ds = PetDataset(train_samples, train_transform)
val_ds   = PetDataset(val_samples,   val_transform)

# num_workers=4 processes data asynchronously so the CPU isn't waiting on IO
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ── ShuffleNet V2 ──────────────────────────────────────────────────────────────
def channel_shuffle(x, groups):
    n, c, h, w = x.size()
    x = x.view(n, groups, c // groups, h, w)
    x = x.transpose(1, 2).contiguous()
    return x.view(n, c, h, w)

class InvertedResidual(nn.Module):
    def __init__(self, in_c, out_c, stride):
        super().__init__()
        self.stride = stride
        branch_c = out_c // 2

        if stride == 1:
            self.branch2 = nn.Sequential(
                nn.Conv2d(branch_c, branch_c, 1, bias=False),
                nn.BatchNorm2d(branch_c), nn.ReLU(inplace=True),
                nn.Conv2d(branch_c, branch_c, 3, padding=1, groups=branch_c, bias=False),
                nn.BatchNorm2d(branch_c),
                nn.Conv2d(branch_c, branch_c, 1, bias=False),
                nn.BatchNorm2d(branch_c), nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential(
                nn.Conv2d(in_c, in_c, 3, stride=2, padding=1, groups=in_c, bias=False),
                nn.BatchNorm2d(in_c),
                nn.Conv2d(in_c, branch_c, 1, bias=False),
                nn.BatchNorm2d(branch_c), nn.ReLU(inplace=True),
            )
            self.branch2 = nn.Sequential(
                nn.Conv2d(in_c, branch_c, 1, bias=False),
                nn.BatchNorm2d(branch_c), nn.ReLU(inplace=True),
                nn.Conv2d(branch_c, branch_c, 3, stride=2, padding=1, groups=branch_c, bias=False),
                nn.BatchNorm2d(branch_c),
                nn.Conv2d(branch_c, branch_c, 1, bias=False),
                nn.BatchNorm2d(branch_c), nn.ReLU(inplace=True),
            )

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        return channel_shuffle(out, 2)

class ShuffleNetV2(nn.Module):
    def __init__(self, num_classes, scale=1.0):
        super().__init__()
        configs = {
            0.5:  [48,  96,  192,  1024],
            1.0:  [116, 232, 464,  1024],
            1.5:  [176, 352, 704,  1024],
            2.0:  [244, 488, 976,  2048],
        }
        c = configs[scale]

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 24, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24), nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.stage2  = self._make_stage(24,   c[0], 4)
        self.stage3  = self._make_stage(c[0], c[1], 8)
        self.stage4  = self._make_stage(c[1], c[2], 4)
        self.conv5   = nn.Sequential(
            nn.Conv2d(c[2], c[3], 1, bias=False),
            nn.BatchNorm2d(c[3]), nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(c[3], num_classes)

    def _make_stage(self, in_c, out_c, repeats):
        layers = [InvertedResidual(in_c, out_c, stride=2)]
        for _ in range(repeats - 1):
            layers.append(InvertedResidual(out_c, out_c, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = x.mean([2, 3]) # Global Average Pooling
        return self.fc(x)

# Gracefully handle 0 classes to prevent errors when testing script without data
num_classes = max(1, len(classes)) 
model = ShuffleNetV2(num_classes=num_classes, scale=1.0)

# Memory format optimization: channels_last is up to 2x faster for CPU CNNs
model = model.to(memory_format=torch.channels_last)

optimizer = torch.optim.SGD(
    model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.CrossEntropyLoss()

# ── Training loop ──────────────────────────────────────────────────────────────
if len(classes) == 0:
    print("No images found, stopping before training loop.")
    exit(0)

training_start = time.time()

best_val_acc  = 0.0
final_val_acc = 0.0
final_val_loss = 0.0

for epoch in range(1, EPOCHS + 1):
    epoch_start = time.time()
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in train_loader:
        # Convert images to channels_last for fast computation
        imgs = imgs.to(memory_format=torch.channels_last)
        
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

    train_acc  = correct / total if total else 0
    train_loss = total_loss / len(train_loader) if len(train_loader) else 0

    # ── Validation ─────────────────────────────────────────────────────────────
    model.eval()
    vc, vt, val_loss_sum = 0, 0, 0.0
    
    # Context manager disables gradient graphs for faster validation
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(memory_format=torch.channels_last)
            logits = model(imgs)
            loss   = criterion(logits, labels)
            
            val_loss_sum += loss.item()
            preds  = logits.argmax(dim=1)
            vc    += (preds == labels).sum().item()
            vt    += labels.size(0)

    val_acc  = vc / vt if vt else 0.0
    val_loss = val_loss_sum / len(val_loader) if len(val_loader) > 0 else 0.0

    # Track best and store final epoch values
    if val_acc > best_val_acc:
        best_val_acc = val_acc
    final_val_acc  = val_acc
    final_val_loss = val_loss

    epoch_time = time.time() - epoch_start
    scheduler.step()

    print(f"Epoch [{epoch:02d}/{EPOCHS}]  "
          f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}  |  "
          f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}  |  "
          f"Time: {epoch_time:.1f}s")

# ── Final summary ──────────────────────────────────────────────────────────────
total_time  = time.time() - training_start
hours       = int(total_time // 3600)
minutes     = int((total_time % 3600) // 60)
seconds     = int(total_time % 60)

print("\n" + "═" * 60)
print("              TRAINING COMPLETE — FINAL RESULTS")
print("═" * 60)
print(f"  Final Validation Accuracy : {final_val_acc * 100:.2f}%")
print(f"  Best  Validation Accuracy : {best_val_acc  * 100:.2f}%")
print(f"  Final Validation Loss     : {final_val_loss:.4f}")
print(f"  Total Training Time       : {hours:02d}h {minutes:02d}m {seconds:02d}s")
print("═" * 60)

# ── Save model ─────────────────────────────────────────────────────────────────
torch.save(model.state_dict(), "shufflenet_pet.pt")
torch.save(optimizer.state_dict(), "shufflenet_pet_opt.pt")
print("Model saved → shufflenet_pet.pt")