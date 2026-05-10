import os
import time
import math
import pickle
from typing import Any

# ── Force CPU BEFORE importing JAX ─────────────────────────────────────────────
os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax

# ── Config ─────────────────────────────────────────────────────────────────────
TRAIN_FILE = "./ECG5000/ECG5000_TRAIN.txt"
TEST_FILE  = "./ECG5000/ECG5000_TEST.txt"
EPOCHS     = 20
BATCH_SIZE = 64
LR         = 1e-3
SEED       = 42
WEIGHT_DECAY = 1e-4

np.random.seed(SEED)

print("Using JAX devices:", jax.devices())

# ── Load ECG5000 .txt (space/tab separated, first column = label) ──────────────
def load_ecg_txt(path):
    data = np.loadtxt(path)
    labels = data[:, 0].astype(np.int32)
    series = data[:, 1:].astype(np.float32)
    labels = labels - 1
    return series, labels

print("Loading data...")
X_train, y_train = load_ecg_txt(TRAIN_FILE)
X_test,  y_test  = load_ecg_txt(TEST_FILE)

NUM_CLASSES = len(np.unique(y_train))
SEQ_LEN     = X_train.shape[1]
INPUT_SIZE  = 1

print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")
print(f"Classes: {NUM_CLASSES}  |  Sequence length: {SEQ_LEN}")

# Reshape once up front to match LSTM input: (N, seq_len, input_size)
X_train = X_train[:, :, np.newaxis]
X_test  = X_test[:, :, np.newaxis]

# ── Data loader in JAX-friendly style ──────────────────────────────────────────
def make_epoch_indices(n, batch_size, shuffle, seed, epoch):
    indices = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed + epoch)
        rng.shuffle(indices)
    return indices

def batch_iterator(X, y, batch_size, shuffle, seed, epoch):
    indices = make_epoch_indices(len(y), batch_size, shuffle, seed, epoch)
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        yield X[batch_idx], y[batch_idx]

def num_batches(n, batch_size):
    return math.ceil(n / batch_size) if n > 0 else 0

# ── LSTM Model ─────────────────────────────────────────────────────────────────
class ECG_LSTM(nn.Module):
    hidden_size: int
    num_layers: int
    num_classes: int
    dropout: float = 0.3

    @nn.compact
    def __call__(self, x, train: bool):
        # x: (batch, seq_len, input_size)
        for i in range(self.num_layers):
            x = nn.RNN(
                nn.LSTMCell(features=self.hidden_size),
                return_carry=False,
                name=f"lstm_{i}",
            )(x)

        x = x[:, -1, :]
        x = nn.Dropout(rate=self.dropout, deterministic=not train)(x)
        x = nn.Dense(self.num_classes, name="fc")(x)
        return x

# ── Train state ────────────────────────────────────────────────────────────────
class TrainState(train_state.TrainState):
    pass

def create_learning_rate_fn(steps_per_epoch):
    return optax.cosine_decay_schedule(
        init_value=LR,
        decay_steps=EPOCHS * steps_per_epoch,
        alpha=0.0,
    )

def create_train_state(rng, model, steps_per_epoch):
    dummy_x = jnp.ones((1, SEQ_LEN, INPUT_SIZE), dtype=jnp.float32)
    variables = model.init(
        {"params": rng, "dropout": jax.random.PRNGKey(SEED + 1)},
        dummy_x,
        train=True,
    )
    params = variables["params"]

    lr_fn = create_learning_rate_fn(steps_per_epoch)
    tx = optax.adamw(
        learning_rate=lr_fn,
        weight_decay=WEIGHT_DECAY,
    )

    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    ), lr_fn

def cross_entropy_loss(logits, labels):
    one_hot = jax.nn.one_hot(labels, logits.shape[-1], dtype=logits.dtype)
    return optax.softmax_cross_entropy(logits, one_hot).mean()

@jax.jit
def train_step(state, x, y, dropout_key):
    def loss_fn(params):
        logits = state.apply_fn(
            {"params": params},
            x,
            train=True,
            rngs={"dropout": dropout_key},
        )
        loss = cross_entropy_loss(logits, y)
        return loss, logits

    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)

    preds = jnp.argmax(logits, axis=-1)
    correct = jnp.sum(preds == y)
    return state, loss, correct

@jax.jit
def eval_step(state, x, y):
    logits = state.apply_fn(
        {"params": state.params},
        x,
        train=False,
    )
    loss = cross_entropy_loss(logits, y)
    preds = jnp.argmax(logits, axis=-1)
    correct = jnp.sum(preds == y)
    return loss, correct

def count_parameters(params):
    return sum(x.size for x in jax.tree_util.tree_leaves(params))

def to_host_tree(tree):
    return jax.tree_util.tree_map(lambda x: np.asarray(x), tree)

# ── Build model/optimizer ──────────────────────────────────────────────────────
steps_per_epoch = max(1, num_batches(len(y_train), BATCH_SIZE))

model = ECG_LSTM(
    hidden_size=128,
    num_layers=2,
    num_classes=NUM_CLASSES,
    dropout=0.3,
)

state, lr_fn = create_train_state(jax.random.PRNGKey(SEED), model, steps_per_epoch)

print(f"\nModel parameters: {count_parameters(state.params):,}")
print("─" * 65)

# ── Training loop ──────────────────────────────────────────────────────────────
training_start = time.time()

best_val_acc   = 0.0
final_val_acc  = 0.0
final_val_loss = 0.0

dropout_master_key = jax.random.PRNGKey(SEED + 123)

for epoch in range(1, EPOCHS + 1):
    epoch_start = time.time()

    # ── Train ──────────────────────────────────────────────────────────────────
    train_loss_sum = 0.0
    correct = 0
    total = 0
    train_batches = 0

    for batch_idx, (X_batch, y_batch) in enumerate(
        batch_iterator(X_train, y_train, BATCH_SIZE, shuffle=True, seed=SEED, epoch=epoch)
    ):
        X_batch = jnp.asarray(X_batch, dtype=jnp.float32)
        y_batch = jnp.asarray(y_batch, dtype=jnp.int32)

        dropout_master_key, dropout_key = jax.random.split(dropout_master_key)
        state, loss, batch_correct = train_step(state, X_batch, y_batch, dropout_key)

        train_loss_sum += float(loss)
        correct += int(batch_correct)
        total += y_batch.shape[0]
        train_batches += 1

    train_acc  = correct / total if total else 0.0
    train_loss = train_loss_sum / train_batches if train_batches else 0.0

    # ── Validation (test set) ──────────────────────────────────────────────────
    vc, vt, val_loss_sum = 0, 0, 0.0
    test_batches = 0

    for X_batch, y_batch in batch_iterator(
        X_test, y_test, BATCH_SIZE, shuffle=False, seed=SEED, epoch=epoch
    ):
        X_batch = jnp.asarray(X_batch, dtype=jnp.float32)
        y_batch = jnp.asarray(y_batch, dtype=jnp.int32)

        loss, batch_correct = eval_step(state, X_batch, y_batch)

        val_loss_sum += float(loss)
        vc += int(batch_correct)
        vt += y_batch.shape[0]
        test_batches += 1

    val_acc  = vc / vt if vt else 0.0
    val_loss = val_loss_sum / test_batches if test_batches else 0.0

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_ckpt = {
            "params": to_host_tree(state.params),
            "config": {
                "input_size": INPUT_SIZE,
                "hidden_size": 128,
                "num_layers": 2,
                "num_classes": NUM_CLASSES,
                "dropout": 0.3,
                "seq_len": SEQ_LEN,
            },
        }
        with open("ecg_lstm_best_flax.pkl", "wb") as f:
            pickle.dump(best_ckpt, f)

    final_val_acc  = val_acc
    final_val_loss = val_loss

    epoch_time = time.time() - epoch_start
    current_lr = float(lr_fn(state.step))

    print(
        f"Epoch [{epoch:02d}/{EPOCHS}]  "
        f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}  |  "
        f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}  |  "
        f"LR: {current_lr:.6f}  "
        f"Time: {epoch_time:.1f}s"
    )

# ── Final summary ──────────────────────────────────────────────────────────────
total_time = time.time() - training_start
hours      = int(total_time // 3600)
minutes    = int((total_time % 3600) // 60)
seconds    = int(total_time % 60)

print("\n" + "═" * 65)
print("               TRAINING COMPLETE — FINAL RESULTS")
print("═" * 65)
print(f"  Final Validation Accuracy : {final_val_acc  * 100:.2f}%")
print(f"  Best  Validation Accuracy : {best_val_acc   * 100:.2f}%")
print(f"  Final Validation Loss     : {final_val_loss:.4f}")
print(f"  Total Training Time       : {hours:02d}h {minutes:02d}m {seconds:02d}s")
print("═" * 65)

# ── Save final model + optimizer state ────────────────────────────────────────
final_model_ckpt = {
    "params": to_host_tree(state.params),
    "config": {
        "input_size": INPUT_SIZE,
        "hidden_size": 128,
        "num_layers": 2,
        "num_classes": NUM_CLASSES,
        "dropout": 0.3,
        "seq_len": SEQ_LEN,
    },
}

final_opt_ckpt = {
    "opt_state": to_host_tree(state.opt_state),
    "step": int(jax.device_get(state.step)),
}

with open("ecg_lstm_final_flax.pkl", "wb") as f:
    pickle.dump(final_model_ckpt, f)

with open("ecg_lstm_final_optax.pkl", "wb") as f:
    pickle.dump(final_opt_ckpt, f)

print("Models saved → ecg_lstm_best_flax.pkl  |  ecg_lstm_final_flax.pkl")