import os
import time
import random
import numpy as np
from PIL import Image
import paddle
import paddle.nn as nn
import paddle.vision.transforms as T
from paddle.io import Dataset, DataLoader

# ── Force CPU ──────────────────────────────────────────────────────────────────
paddle.set_device("cpu")

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
paddle.seed(SEED)

# ── Build class list from image filenames ──────────────────────────────────────
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
        return img, np.int64(label)

train_ds = PetDataset(train_samples, train_transform)
val_ds   = PetDataset(val_samples,   val_transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ── ShuffleNet V2 ──────────────────────────────────────────────────────────────
def channel_shuffle(x, groups):
    n, c, h, w = x.shape
    x = x.reshape([n, groups, c // groups, h, w])
    x = x.transpose([0, 2, 1, 3, 4])
    return x.reshape([n, c, h, w])

class InvertedResidual(nn.Layer):
    def __init__(self, in_c, out_c, stride):
        super().__init__()
        self.stride   = stride
        branch_c      = out_c // 2

        if stride == 1:
            self.branch2 = nn.Sequential(
                nn.Conv2D(branch_c, branch_c, 1, bias_attr=False),
                nn.BatchNorm2D(branch_c), nn.ReLU(),
                nn.Conv2D(branch_c, branch_c, 3, padding=1,
                          groups=branch_c, bias_attr=False),
                nn.BatchNorm2D(branch_c),
                nn.Conv2D(branch_c, branch_c, 1, bias_attr=False),
                nn.BatchNorm2D(branch_c), nn.ReLU(),
            )
        else:
            self.branch1 = nn.Sequential(
                nn.Conv2D(in_c, in_c, 3, stride=2, padding=1,
                          groups=in_c, bias_attr=False),
                nn.BatchNorm2D(in_c),
                nn.Conv2D(in_c, branch_c, 1, bias_attr=False),
                nn.BatchNorm2D(branch_c), nn.ReLU(),
            )
            self.branch2 = nn.Sequential(
                nn.Conv2D(in_c, branch_c, 1, bias_attr=False),
                nn.BatchNorm2D(branch_c), nn.ReLU(),
                nn.Conv2D(branch_c, branch_c, 3, stride=2, padding=1,
                          groups=branch_c, bias_attr=False),
                nn.BatchNorm2D(branch_c),
                nn.Conv2D(branch_c, branch_c, 1, bias_attr=False),
                nn.BatchNorm2D(branch_c), nn.ReLU(),
            )

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.split(2, axis=1)
            out = paddle.concat([x1, self.branch2(x2)], axis=1)
        else:
            out = paddle.concat([self.branch1(x), self.branch2(x)], axis=1)
        return channel_shuffle(out, 2)

class ShuffleNetV2(nn.Layer):
    def __init__(self, num_classes, scale=1.0):
        super().__init__()
        configs = {
            0.5:  [48,  96,  192,  1024],
            1.0:  [116, 232, 464,  1024],
            1.5:  [176, 352, 704,  1024],
            2.0:  [244, 488, 976,  2048],
        }
        c = configs[scale]

        self.conv1   = nn.Sequential(
            nn.Conv2D(3, 24, 3, stride=2, padding=1, bias_attr=False),
            nn.BatchNorm2D(24), nn.ReLU()
        )
        self.maxpool = nn.MaxPool2D(3, stride=2, padding=1)
        self.stage2  = self._make_stage(24,   c[0], 4)
        self.stage3  = self._make_stage(c[0], c[1], 8)
        self.stage4  = self._make_stage(c[1], c[2], 4)
        self.conv5   = nn.Sequential(
            nn.Conv2D(c[2], c[3], 1, bias_attr=False),
            nn.BatchNorm2D(c[3]), nn.ReLU()
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
        x = paddle.mean(x, axis=[2, 3])
        return self.fc(x)

model     = ShuffleNetV2(num_classes=len(classes), scale=1.0)
optimizer = paddle.optimizer.Momentum(
    learning_rate=LR, momentum=0.9,
    parameters=model.parameters(),
    weight_decay=1e-4
)
scheduler = paddle.optimizer.lr.CosineAnnealingDecay(LR, T_max=EPOCHS)
criterion = nn.CrossEntropyLoss()

# ── Training loop ──────────────────────────────────────────────────────────────
training_start = time.time()

best_val_acc  = 0.0
final_val_acc = 0.0
final_val_loss = 0.0

for epoch in range(1, EPOCHS + 1):
    epoch_start = time.time()
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in train_loader:
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()

        total_loss += float(loss)
        preds       = paddle.argmax(logits, axis=1)
        correct    += int((preds == labels).sum())
        total      += labels.shape[0]

    train_acc  = correct / total
    train_loss = total_loss / len(train_loader)

    # ── Validation ─────────────────────────────────────────────────────────────
    model.eval()
    vc, vt, val_loss_sum = 0, 0, 0.0
    with paddle.no_grad():
        for imgs, labels in val_loader:
            logits = model(imgs)
            loss   = criterion(logits, labels)
            val_loss_sum += float(loss)
            preds  = paddle.argmax(logits, axis=1)
            vc    += int((preds == labels).sum())
            vt    += labels.shape[0]

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
paddle.save(model.state_dict(), "shufflenet_pet.pdparams")
paddle.save(optimizer.state_dict(), "shufflenet_pet_opt.pdopt")
print("Model saved → shufflenet_pet.pdparams")