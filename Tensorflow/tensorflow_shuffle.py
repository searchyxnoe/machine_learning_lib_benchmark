import os
import time
import random
import numpy as np

# ── Force CPU before importing TensorFlow ──────────────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

# Optimize CPU threads for parallel execution
tf.config.threading.set_inter_op_parallelism_threads(0)
tf.config.threading.set_intra_op_parallelism_threads(0)

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
tf.random.set_seed(SEED)

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

train_paths = [s[0] for s in train_samples]
train_labels = [s[1] for s in train_samples]
val_paths = [s[0] for s in val_samples]
val_labels = [s[1] for s in val_samples]

# ── Transforms & Dataset Pipeline ──────────────────────────────────────────────
MEAN = tf.constant([0.485, 0.456, 0.406])
STD  = tf.constant([0.229, 0.224, 0.225])

def parse_image(file_path, label):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32) / 255.0  # Scale to [0, 1]
    return img, label

def train_augment(img, label):
    img = tf.image.resize(img, [IMG_SIZE + 32, IMG_SIZE + 32])
    img = tf.image.random_crop(img, size=[IMG_SIZE, IMG_SIZE, 3])
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.2)
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
    img = tf.image.random_saturation(img, lower=0.8, upper=1.2)
    img = tf.clip_by_value(img, 0.0, 1.0)
    img = (img - MEAN) / STD
    return img, label

def val_augment(img, label):
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = (img - MEAN) / STD
    return img, label

AUTOTUNE = tf.data.AUTOTUNE

train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
train_ds = train_ds.shuffle(len(train_paths), seed=SEED)
train_ds = train_ds.map(parse_image, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.map(train_augment, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
val_ds = val_ds.map(parse_image, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(val_augment, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

# ── ShuffleNet V2 ──────────────────────────────────────────────────────────────
def channel_shuffle(x, groups):
    # Operating on NHWC for optimal CPU efficiency
    batch_size = tf.shape(x)[0]
    height, width, channels = tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
    channels_per_group = channels // groups
    
    x = tf.reshape(x, [batch_size, height, width, groups, channels_per_group])
    x = tf.transpose(x, [0, 1, 2, 4, 3])
    x = tf.reshape(x, [batch_size, height, width, channels])
    return x

class InvertedResidual(tf.keras.layers.Layer):
    def __init__(self, in_c, out_c, stride, **kwargs):
        super().__init__(**kwargs)
        self.stride = stride
        branch_c = out_c // 2

        if stride == 1:
            self.branch2 = tf.keras.Sequential([
                tf.keras.layers.Conv2D(branch_c, 1, use_bias=False),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.DepthwiseConv2D(3, padding='same', use_bias=False),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(branch_c, 1, use_bias=False),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU()
            ])
        else:
            self.branch1 = tf.keras.Sequential([
                tf.keras.layers.DepthwiseConv2D(3, strides=2, padding='same', use_bias=False),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(branch_c, 1, use_bias=False),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU()
            ])
            self.branch2 = tf.keras.Sequential([
                tf.keras.layers.Conv2D(branch_c, 1, use_bias=False),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.DepthwiseConv2D(3, strides=2, padding='same', use_bias=False),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(branch_c, 1, use_bias=False),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU()
            ])

    def call(self, x, training=False):
        if self.stride == 1:
            x1, x2 = tf.split(x, num_or_size_splits=2, axis=-1)
            out = tf.concat([x1, self.branch2(x2, training=training)], axis=-1)
        else:
            out = tf.concat([self.branch1(x, training=training), self.branch2(x, training=training)], axis=-1)
        return channel_shuffle(out, 2)

class ShuffleNetV2(tf.keras.Model):
    def __init__(self, num_classes, scale=1.0, **kwargs):
        super().__init__(**kwargs)
        configs = {
            0.5:  [48,  96,  192,  1024],
            1.0:  [116, 232, 464,  1024],
            1.5:  [176, 352, 704,  1024],
            2.0:  [244, 488, 976,  2048],
        }
        c = configs[scale]

        self.conv1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(24, 3, strides=2, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ])
        self.maxpool = tf.keras.layers.MaxPooling2D(3, strides=2, padding='same')
        
        self.stage2 = self._make_stage(24,   c[0], 4)
        self.stage3 = self._make_stage(c[0], c[1], 8)
        self.stage4 = self._make_stage(c[1], c[2], 4)
        
        self.conv5 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(c[3], 1, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ])
        self.gap = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(num_classes)

    def _make_stage(self, in_c, out_c, repeats):
        layers = [InvertedResidual(in_c, out_c, stride=2)]
        for _ in range(repeats - 1):
            layers.append(InvertedResidual(out_c, out_c, stride=1))
        return tf.keras.Sequential(layers)

    def call(self, x, training=False):
        x = self.conv1(x, training=training)
        x = self.maxpool(x)
        x = self.stage2(x, training=training)
        x = self.stage3(x, training=training)
        x = self.stage4(x, training=training)
        x = self.conv5(x, training=training)
        x = self.gap(x)
        return self.fc(x)

# ── Initialization and Compilation ──────────────────────────────────────────────
model = ShuffleNetV2(num_classes=len(classes), scale=1.0)
steps_per_epoch = int(np.ceil(len(train_samples) / BATCH_SIZE))
total_steps = EPOCHS * steps_per_epoch

# TF idiomatic smooth cosine decay across steps rather than abrupt epoch shifts
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=LR, 
    decay_steps=total_steps
)

optimizer = tf.keras.optimizers.SGD(
    learning_rate=lr_schedule, 
    momentum=0.9, 
    weight_decay=1e-4
)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# ── Accelerated Graph Step Functions ───────────────────────────────────────────
@tf.function
def train_step(imgs, labels):
    with tf.GradientTape() as tape:
        logits = model(imgs, training=True)
        loss = loss_fn(labels, logits)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    preds = tf.argmax(logits, axis=1, output_type=tf.int32)
    correct = tf.reduce_sum(tf.cast(preds == labels, tf.int32))
    return loss, correct

@tf.function
def val_step(imgs, labels):
    logits = model(imgs, training=False)
    loss = loss_fn(labels, logits)
    
    preds = tf.argmax(logits, axis=1, output_type=tf.int32)
    correct = tf.reduce_sum(tf.cast(preds == labels, tf.int32))
    return loss, correct

# ── Training loop ──────────────────────────────────────────────────────────────
training_start = time.time()

best_val_acc  = 0.0
final_val_acc = 0.0
final_val_loss = 0.0

for epoch in range(1, EPOCHS + 1):
    epoch_start = time.time()
    
    # Train
    total_loss, correct, total, batches = 0.0, 0, 0, 0
    for imgs, labels in train_ds:
        loss, corr = train_step(imgs, labels)
        total_loss += float(loss)
        correct    += int(corr)
        total      += int(tf.shape(labels)[0])
        batches    += 1

    train_acc  = correct / total if total > 0 else 0.0
    train_loss = total_loss / batches if batches > 0 else 0.0

    # Validate
    vc, vt, val_loss_sum, val_batches = 0, 0, 0.0, 0
    for imgs, labels in val_ds:
        loss, corr = val_step(imgs, labels)
        val_loss_sum += float(loss)
        vc           += int(corr)
        vt           += int(tf.shape(labels)[0])
        val_batches  += 1

    val_acc  = vc / vt if vt > 0 else 0.0
    val_loss = val_loss_sum / val_batches if val_batches > 0 else 0.0

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        
    final_val_acc  = val_acc
    final_val_loss = val_loss
    epoch_time = time.time() - epoch_start

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
model.save_weights("shufflenet_pet.weights.h5")
print("Model saved → shufflenet_pet.weights.h5")