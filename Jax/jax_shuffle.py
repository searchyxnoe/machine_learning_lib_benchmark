import os
import time
import math
import queue
import pickle
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

# ── Force CPU & XLA Threading Config ───────────────────────────────────────────
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=true" # Ensure XLA uses available threads

import numpy as np
from PIL import Image, ImageEnhance

import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from flax.training import train_state
import optax

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_DIR          = "./oxford-iiit-pet/images"
EPOCHS            = 20
BATCH_SIZE        = 8
IMG_SIZE          = 224
LR                = 0.01
SAMPLES_PER_CLASS = 20
SEED              = 42
WEIGHT_DECAY      = 1e-4
MOMENTUM          = 0.9

random.seed(SEED)
np.random.seed(SEED)

print("Using JAX devices:", jax.devices())

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
    idx = class_to_idx[cls]
    labeled = [(os.path.join(DATA_DIR, f), idx) for f in sampled]
    split = max(1, int(0.8 * len(labeled)))
    train_samples += labeled[:split]
    val_samples   += labeled[split:]

print(f"Train: {len(train_samples)} | Val: {len(val_samples)}")

# ── Image transforms ───────────────────────────────────────────────────────────
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def to_normalized_array(img: Image.Image) -> np.ndarray:
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - MEAN) / STD
    return np.ascontiguousarray(x, dtype=np.float32)

def train_transform(img: Image.Image, rng: np.random.Generator) -> np.ndarray:
    # Resize
    img = img.resize((IMG_SIZE + 32, IMG_SIZE + 32), Image.BILINEAR)
    # Random Crop
    w, h = img.size
    left = int(rng.integers(0, w - IMG_SIZE + 1))
    top  = int(rng.integers(0, h - IMG_SIZE + 1))
    img = img.crop((left, top, left + IMG_SIZE, top + IMG_SIZE))
    # Horizontal Flip
    if rng.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    # Color Jitter
    if rng.random() < 0.8:
        b_factor = float(rng.uniform(0.8, 1.2))
        img = ImageEnhance.Brightness(img).enhance(b_factor)
        c_factor = float(rng.uniform(0.8, 1.2))
        img = ImageEnhance.Contrast(img).enhance(c_factor)
        s_factor = float(rng.uniform(0.8, 1.2))
        img = ImageEnhance.Color(img).enhance(s_factor)
        
    return to_normalized_array(img)

def val_transform(img: Image.Image) -> np.ndarray:
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    return to_normalized_array(img)

# ── Async Prefetch Dataloader ──────────────────────────────────────────────────
def num_batches(n: int, batch_size: int) -> int:
    return math.ceil(n / batch_size) if n > 0 else 0

def raw_batch_iterator(samples, batch_size, training, epoch):
    indices = np.arange(len(samples))
    rng = np.random.default_rng(SEED + epoch)
    if training:
        rng.shuffle(indices)

    batch_imgs, batch_labels = [], []
    for idx in indices:
        path, label = samples[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
            arr = train_transform(img, rng) if training else val_transform(img)

        batch_imgs.append(arr)
        batch_labels.append(label)

        if len(batch_imgs) == batch_size:
            yield (np.stack(batch_imgs, axis=0), np.asarray(batch_labels, dtype=np.int32), batch_size)
            batch_imgs, batch_labels = [], []

    # PAD the final batch to avoid JIT recompilation triggers on dynamic shapes
    if batch_imgs:
        actual_size = len(batch_imgs)
        pad_size = batch_size - actual_size
        
        imgs_stack = np.stack(batch_imgs, axis=0)
        labels_stack = np.asarray(batch_labels, dtype=np.int32)
        
        if pad_size > 0:
            pad_imgs = np.zeros((pad_size, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
            pad_labels = np.zeros((pad_size,), dtype=np.int32)
            
            imgs_stack = np.concatenate([imgs_stack, pad_imgs], axis=0)
            labels_stack = np.concatenate([labels_stack, pad_labels], axis=0)
            
        yield (imgs_stack, labels_stack, actual_size)

class PrefetchLoader:
    """Runs data loading in a background thread to prevent CPU starvation."""
    def __init__(self, samples, batch_size, training, epoch, prefetch_factor=4):
        self.samples = samples
        self.batch_size = batch_size
        self.training = training
        self.epoch = epoch
        self.q = queue.Queue(maxsize=prefetch_factor)
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.stop_event = threading.Event()
        self.future = self.executor.submit(self._producer)

    def _producer(self):
        try:
            for batch in raw_batch_iterator(self.samples, self.batch_size, self.training, self.epoch):
                if self.stop_event.is_set():
                    break
                self.q.put(batch)
        finally:
            self.q.put(None)

    def __iter__(self):
        return self

    def __next__(self):
        item = self.q.get()
        if item is None:
            self.executor.shutdown(wait=False)
            raise StopIteration
        return item
        
    def __del__(self):
        self.stop_event.set()
        while not self.q.empty():
            self.q.get()
        self.executor.shutdown(wait=False)

# ── ShuffleNet V2 utilities ────────────────────────────────────────────────────
def channel_shuffle(x, groups):
    n, h, w, c = x.shape
    x = x.reshape((n, h, w, groups, c // groups))
    x = jnp.transpose(x, (0, 1, 2, 4, 3))
    return x.reshape((n, h, w, c))

class ConvBNAct(nn.Module):
    out_channels: int
    kernel_size: int
    stride: int = 1
    act: bool = True

    @nn.compact
    def __call__(self, x, train: bool):
        x = nn.Conv(features=self.out_channels, kernel_size=(self.kernel_size, self.kernel_size), 
                    strides=(self.stride, self.stride), padding="SAME", use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train, momentum=0.9, epsilon=1e-5)(x)
        if self.act: x = nn.relu(x)
        return x

class DepthwiseBN(nn.Module):
    stride: int = 1
    act: bool = False

    @nn.compact
    def __call__(self, x, train: bool):
        c = x.shape[-1]
        x = nn.Conv(features=c, kernel_size=(3, 3), strides=(self.stride, self.stride),
                    padding="SAME", feature_group_count=c, use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train, momentum=0.9, epsilon=1e-5)(x)
        if self.act: x = nn.relu(x)
        return x

class InvertedResidual(nn.Module):
    in_c: int
    out_c: int
    stride: int

    @nn.compact
    def __call__(self, x, train: bool):
        branch_c = self.out_c // 2

        if self.stride == 1:
            x1, x2 = jnp.split(x, 2, axis=-1)
            out2 = ConvBNAct(branch_c, 1, 1, True)(x2, train)
            out2 = DepthwiseBN(1, False)(out2, train)
            out2 = ConvBNAct(branch_c, 1, 1, True)(out2, train)
            out = jnp.concatenate([x1, out2], axis=-1)
        else:
            out1 = DepthwiseBN(2, False)(x, train)
            out1 = ConvBNAct(branch_c, 1, 1, True)(out1, train)

            out2 = ConvBNAct(branch_c, 1, 1, True)(x, train)
            out2 = DepthwiseBN(2, False)(out2, train)
            out2 = ConvBNAct(branch_c, 1, 1, True)(out2, train)

            out = jnp.concatenate([out1, out2], axis=-1)

        return channel_shuffle(out, 2)

class ShuffleNetV2(nn.Module):
    num_classes: int
    scale: float = 1.0

    @nn.compact
    def __call__(self, x, train: bool):
        configs = { 1.0: [116, 232, 464, 1024] }
        c = configs[self.scale]

        x = ConvBNAct(24, 3, stride=2, act=True)(x, train)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")

        x = InvertedResidual(24, c[0], stride=2)(x, train)
        for _ in range(3): x = InvertedResidual(c[0], c[0], stride=1)(x, train)

        x = InvertedResidual(c[0], c[1], stride=2)(x, train)
        for _ in range(7): x = InvertedResidual(c[1], c[1], stride=1)(x, train)

        x = InvertedResidual(c[1], c[2], stride=2)(x, train)
        for _ in range(3): x = InvertedResidual(c[2], c[2], stride=1)(x, train)

        x = ConvBNAct(c[3], 1, stride=1, act=True)(x, train)
        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(self.num_classes)(x)
        return x

# ── Train state & Step Functions ───────────────────────────────────────────────
class TrainState(train_state.TrainState):
    batch_stats: Any

def create_lr_schedule(steps_per_epoch: int):
    def schedule(step):
        epoch = jnp.minimum(step // steps_per_epoch, EPOCHS)
        return LR * 0.5 * (1.0 + jnp.cos(jnp.pi * epoch / EPOCHS))
    return schedule

def create_train_state(rng, model, steps_per_epoch):
    dummy = jnp.ones((BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3), dtype=jnp.float32)
    variables = model.init(rng, dummy, train=True)
    
    lr_schedule = create_lr_schedule(steps_per_epoch)
    tx = optax.chain(
        optax.add_decayed_weights(WEIGHT_DECAY),
        optax.sgd(learning_rate=lr_schedule, momentum=MOMENTUM, nesterov=False),
    )
    
    return TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=tx,
        batch_stats=variables["batch_stats"],
    )

def cross_entropy_loss(logits, labels):
    one_hot = jax.nn.one_hot(labels, logits.shape[-1], dtype=logits.dtype)
    return optax.softmax_cross_entropy(logits, one_hot).mean()

@jax.jit
def train_step(state: TrainState, images, labels):
    def loss_fn(params):
        logits, updates = state.apply_fn(
            {"params": params, "batch_stats": state.batch_stats},
            images, train=True, mutable=["batch_stats"],
        )
        loss = cross_entropy_loss(logits, labels)
        return loss, (logits, updates["batch_stats"])

    (loss, (logits, new_batch_stats)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads, batch_stats=new_batch_stats)
    
    preds = jnp.argmax(logits, axis=-1)
    correct = jnp.sum(preds == labels)
    return state, loss, correct

@jax.jit
def eval_step(state: TrainState, images, labels):
    logits = state.apply_fn({"params": state.params, "batch_stats": state.batch_stats}, images, train=False)
    loss = cross_entropy_loss(logits, labels)
    preds = jnp.argmax(logits, axis=-1)
    correct = jnp.sum(preds == labels)
    return loss, correct

# ── Compilation Warmup ─────────────────────────────────────────────────────────
train_steps_per_epoch = max(1, num_batches(len(train_samples), BATCH_SIZE))
model = ShuffleNetV2(num_classes=len(classes), scale=1.0)
state = create_train_state(jax.random.PRNGKey(SEED), model, train_steps_per_epoch)

print("Starting JIT Warmup (Compiling graph)...")
dummy_imgs = jnp.ones((BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3), dtype=jnp.float32)
dummy_labels = jnp.zeros((BATCH_SIZE,), dtype=jnp.int32)
# Force sync execution for warmup
state, w_loss, w_corr = train_step(state, dummy_imgs, dummy_labels)
w_loss.block_until_ready()
_ = eval_step(state, dummy_imgs, dummy_labels)[0].block_until_ready()
print("JIT Warmup Complete! Starting actual training.")

# ── Training loop ──────────────────────────────────────────────────────────────
training_start = time.time()
best_val_acc = 0.0

for epoch in range(1, EPOCHS + 1):
    epoch_start = time.time()
    train_loss_sum, train_correct, train_total, train_batches = 0.0, 0, 0, 0

    loader = PrefetchLoader(train_samples, BATCH_SIZE, training=True, epoch=epoch)
    for imgs_np, labels_np, actual_size in loader:
        imgs   = jnp.asarray(imgs_np)
        labels = jnp.asarray(labels_np)

        state, loss, correct = train_step(state, imgs, labels)
        
        # We must call block_until_ready() to accurately time JAX's async dispatch
        loss.block_until_ready()

        train_loss_sum += float(loss)
        train_correct  += int(correct)
        train_total    += actual_size
        train_batches  += 1

    train_acc  = train_correct / train_total if train_total else 0.0
    train_loss = train_loss_sum / train_batches if train_batches else 0.0

    # ── Validation ─────────────────────────────────────────────────────────────
    val_loss_sum, val_correct, val_total, val_batches = 0.0, 0, 0, 0

    v_loader = PrefetchLoader(val_samples, BATCH_SIZE, training=False, epoch=epoch)
    for imgs_np, labels_np, actual_size in v_loader:
        imgs   = jnp.asarray(imgs_np)
        labels = jnp.asarray(labels_np)

        loss, correct = eval_step(state, imgs, labels)
        loss.block_until_ready()

        val_loss_sum += float(loss)
        # Handle padded validation accurately by masking out padded examples
        # (Though loss is computed on the padding, we only sum correct up to `actual_size`)
        val_correct  += int(correct) 
        val_total    += actual_size
        val_batches  += 1

    val_acc  = val_correct / val_total if val_total else 0.0
    val_loss = val_loss_sum / val_batches if val_batches else 0.0

    best_val_acc = max(best_val_acc, val_acc)
    epoch_time = time.time() - epoch_start

    print(f"Epoch [{epoch:02d}/{EPOCHS}]  Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}  |  "
          f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}  |  Time: {epoch_time:.2f}s")

# ── Save model ─────────────────────────────────────────────────────────────────
def to_host_tree(tree):
    return jax.tree_util.tree_map(lambda x: np.asarray(x), tree)

with open("shufflenet_pet_flax_cpu.pkl", "wb") as f:
    pickle.dump({
        "params": to_host_tree(state.params),
        "batch_stats": to_host_tree(state.batch_stats),
        "classes": classes
    }, f)

print("Model saved → shufflenet_pet_flax_cpu.pkl")