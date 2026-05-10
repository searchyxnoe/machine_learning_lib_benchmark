import os
import time
import random
import numpy as np
from PIL import Image

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, context, set_seed
from mindspore import save_checkpoint, load_checkpoint, load_param_into_net
from mindspore.experimental import optim as ms_optim
import mindspore.dataset as ds
import mindspore.dataset.transforms as C
import mindspore.dataset.vision as vision

# ── Force CPU + Static Graph (fastest for CPU inference/training) ───────────────
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

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
set_seed(SEED)

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

# ── MindSpore GeneratorDataset source ─────────────────────────────────────────
class PetDatasetGenerator:
    """Pure Python iterable; MindSpore's pipeline handles batching & augmentation."""
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
        return img, np.int32(label)

# ── Build MindSpore datasets with vision transforms ───────────────────────────
mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
std  = [0.229 * 255, 0.224 * 255, 0.225 * 255]

def make_dataset(samples, is_train, batch_size, shuffle):
    gen = PetDatasetGenerator(samples)
    dataset = ds.GeneratorDataset(
        gen,
        column_names=["image", "label"],
        shuffle=shuffle,
        num_parallel_workers=1,
        python_multiprocessing=False,
    )
    if is_train:
        trans = [
            vision.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
            vision.RandomCrop(IMG_SIZE),
            vision.RandomHorizontalFlip(),
            vision.RandomColorAdjust(brightness=0.2, contrast=0.2, saturation=0.2),
            vision.ToTensor(),          # HWC uint8 → CHW float32 in [0,1]
            vision.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225],
                             is_hwc=False),
        ]
    else:
        trans = [
            vision.Resize((IMG_SIZE, IMG_SIZE)),
            vision.ToTensor(),
            vision.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225],
                             is_hwc=False),
        ]
    label_trans = [C.TypeCast(mindspore.int32)]
    dataset = dataset.map(operations=trans,       input_columns=["image"])
    dataset = dataset.map(operations=label_trans, input_columns=["label"])
    dataset = dataset.batch(batch_size, drop_remainder=False)
    return dataset

train_dataset = make_dataset(train_samples, is_train=True,  batch_size=BATCH_SIZE, shuffle=True)
val_dataset   = make_dataset(val_samples,   is_train=False, batch_size=BATCH_SIZE, shuffle=False)

# ── ShuffleNet V2 ──────────────────────────────────────────────────────────────
class ChannelShuffle(nn.Cell):
    """Compiled into the static graph; zero Python overhead at runtime."""
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def construct(self, x):
        n, c, h, w = x.shape
        g = self.groups
        x = x.reshape(n, g, c // g, h, w)
        x = x.transpose(0, 2, 1, 3, 4)
        return x.reshape(n, c, h, w)


class InvertedResidual(nn.Cell):
    def __init__(self, in_c, out_c, stride):
        super().__init__()
        self.stride  = stride
        branch_c     = out_c // 2
        self.shuffle = ChannelShuffle(2)

        if stride == 1:
            self.branch2 = nn.SequentialCell(
                nn.Conv2d(branch_c, branch_c, 1, has_bias=False),
                nn.BatchNorm2d(branch_c), nn.ReLU(),
                nn.Conv2d(branch_c, branch_c, 3, pad_mode="pad", padding=1,
                          group=branch_c, has_bias=False),
                nn.BatchNorm2d(branch_c),
                nn.Conv2d(branch_c, branch_c, 1, has_bias=False),
                nn.BatchNorm2d(branch_c), nn.ReLU(),
            )
        else:
            self.branch1 = nn.SequentialCell(
                nn.Conv2d(in_c, in_c, 3, stride=2, pad_mode="pad", padding=1,
                          group=in_c, has_bias=False),
                nn.BatchNorm2d(in_c),
                nn.Conv2d(in_c, branch_c, 1, has_bias=False),
                nn.BatchNorm2d(branch_c), nn.ReLU(),
            )
            self.branch2 = nn.SequentialCell(
                nn.Conv2d(in_c, branch_c, 1, has_bias=False),
                nn.BatchNorm2d(branch_c), nn.ReLU(),
                nn.Conv2d(branch_c, branch_c, 3, stride=2, pad_mode="pad", padding=1,
                          group=branch_c, has_bias=False),
                nn.BatchNorm2d(branch_c),
                nn.Conv2d(branch_c, branch_c, 1, has_bias=False),
                nn.BatchNorm2d(branch_c), nn.ReLU(),
            )

    def construct(self, x):
        if self.stride == 1:
            # split along channel axis
            c = x.shape[1] // 2
            x1 = x[:, :c, :, :]
            x2 = x[:, c:, :, :]
            out = ops.concat((x1, self.branch2(x2)), axis=1)
        else:
            out = ops.concat((self.branch1(x), self.branch2(x)), axis=1)
        return self.shuffle(out)


class ShuffleNetV2(nn.Cell):
    def __init__(self, num_classes, scale=1.0):
        super().__init__()
        configs = {
            0.5:  [48,  96,  192,  1024],
            1.0:  [116, 232, 464,  1024],
            1.5:  [176, 352, 704,  1024],
            2.0:  [244, 488, 976,  2048],
        }
        c = configs[scale]

        self.conv1 = nn.SequentialCell(
            nn.Conv2d(3, 24, 3, stride=2, pad_mode="pad", padding=1, has_bias=False),
            nn.BatchNorm2d(24), nn.ReLU()
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")
        self.stage2  = self._make_stage(24,   c[0], 4)
        self.stage3  = self._make_stage(c[0], c[1], 8)
        self.stage4  = self._make_stage(c[1], c[2], 4)
        self.conv5   = nn.SequentialCell(
            nn.Conv2d(c[2], c[3], 1, has_bias=False),
            nn.BatchNorm2d(c[3]), nn.ReLU()
        )
        self.fc = nn.Dense(c[3], num_classes)
        self.mean = ops.ReduceMean(keep_dims=False)

    def _make_stage(self, in_c, out_c, repeats):
        layers = [InvertedResidual(in_c, out_c, stride=2)]
        for _ in range(repeats - 1):
            layers.append(InvertedResidual(out_c, out_c, stride=1))
        return nn.SequentialCell(*layers)

    def construct(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = self.mean(x, (2, 3))   # Global Average Pooling
        return self.fc(x)

# ── Model, optimizer, scheduler, loss ─────────────────────────────────────────
model     = ShuffleNetV2(num_classes=len(classes), scale=1.0)
optimizer = ms_optim.SGD(
    model.trainable_params(),
    lr=LR, momentum=0.9, weight_decay=1e-4
)
scheduler = ms_optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.CrossEntropyLoss()

# ── Forward + loss cell (compiled as one graph for efficiency) ─────────────────
class TrainStepCell(nn.Cell):
    """Fuses forward pass + loss + backward + update into one compiled graph."""
    def __init__(self, network, loss_fn, optimizer):
        super().__init__()
        self.network   = network
        self.loss_fn   = loss_fn
        self.optimizer = optimizer
        self.grad_fn   = mindspore.value_and_grad(
            self._forward, None, optimizer.parameters
        )

    def _forward(self, imgs, labels):
        logits = self.network(imgs)
        return self.loss_fn(logits, labels)

    def construct(self, imgs, labels):
        loss, grads = self.grad_fn(imgs, labels)
        self.optimizer(grads)
        return loss

train_cell = TrainStepCell(model, criterion, optimizer)

# ── Training loop ──────────────────────────────────────────────────────────────
training_start = time.time()
best_val_acc   = 0.0
final_val_acc  = 0.0
final_val_loss = 0.0

argmax = ops.Argmax(axis=1)
equal  = ops.Equal()
cast   = ops.Cast()

for epoch in range(1, EPOCHS + 1):
    epoch_start = time.time()

    # ── Train ──────────────────────────────────────────────────────────────────
    model.set_train(True)
    total_loss, correct, total = 0.0, 0, 0
    train_steps = 0

    for imgs, labels in train_dataset.create_tuple_iterator():
        loss   = train_cell(imgs, labels)
        logits = model(imgs)                        # forward for accuracy
        preds  = argmax(logits)
        correct   += int(cast(equal(preds, labels), mindspore.float32).sum().asnumpy())
        total     += labels.shape[0]
        total_loss += float(loss.asnumpy())
        train_steps += 1

    train_acc  = correct / total
    train_loss = total_loss / train_steps

    # ── Validation ─────────────────────────────────────────────────────────────
    model.set_train(False)
    vc, vt, val_loss_sum, val_steps = 0, 0, 0.0, 0

    for imgs, labels in val_dataset.create_tuple_iterator():
        logits = model(imgs)
        loss   = criterion(logits, labels)
        val_loss_sum += float(loss.asnumpy())
        preds  = argmax(logits)
        vc    += int(cast(equal(preds, labels), mindspore.float32).sum().asnumpy())
        vt    += labels.shape[0]
        val_steps += 1

    val_acc  = vc / vt if vt else 0.0
    val_loss = val_loss_sum / val_steps if val_steps > 0 else 0.0

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
total_time = time.time() - training_start
hours      = int(total_time // 3600)
minutes    = int((total_time % 3600) // 60)
seconds    = int(total_time % 60)

print("\n" + "═" * 60)
print("              TRAINING COMPLETE — FINAL RESULTS")
print("═" * 60)
print(f"  Final Validation Accuracy : {final_val_acc * 100:.2f}%")
print(f"  Best  Validation Accuracy : {best_val_acc  * 100:.2f}%")
print(f"  Final Validation Loss     : {final_val_loss:.4f}")
print(f"  Total Training Time       : {hours:02d}h {minutes:02d}m {seconds:02d}s")
print("═" * 60)

# ── Save model ─────────────────────────────────────────────────────────────────
save_checkpoint(model, "shufflenet_pet.ckpt")
print("Model saved → shufflenet_pet.ckpt")