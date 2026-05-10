import os
import time
import numpy as np
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, context, set_seed
from mindspore import save_checkpoint, load_checkpoint, load_param_into_net
from mindspore.experimental import optim as ms_optim
import mindspore.dataset as ds

# ── Force CPU + Static Graph ───────────────────────────────────────────────────
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

# ── Config ─────────────────────────────────────────────────────────────────────
TRAIN_FILE = "./ECG5000/ECG5000_TRAIN.txt"
TEST_FILE  = "./ECG5000/ECG5000_TEST.txt"
EPOCHS     = 20
BATCH_SIZE = 64
LR         = 1e-3
SEED       = 42

set_seed(SEED)
np.random.seed(SEED)

# ── Load ECG5000 .txt (space/tab separated, first column = label) ──────────────
def load_ecg_txt(path):
    data   = np.loadtxt(path)                        # (N, 141): col0=label, col1..140=signal
    labels = data[:, 0].astype(np.int32) - 1        # 1-indexed → 0-indexed, int32 for MS
    series = data[:, 1:].astype(np.float32)
    return series, labels

print("Loading data...")
X_train, y_train = load_ecg_txt(TRAIN_FILE)
X_test,  y_test  = load_ecg_txt(TEST_FILE)

NUM_CLASSES = int(np.unique(y_train).shape[0])
SEQ_LEN     = X_train.shape[1]          # 140 time steps
INPUT_SIZE  = 1                          # univariate ECG

print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")
print(f"Classes: {NUM_CLASSES}  |  Sequence length: {SEQ_LEN}")

# ── MindSpore GeneratorDataset ─────────────────────────────────────────────────
class ECGGenerator:
    """Yields (seq, label); MindSpore's C++ pipeline handles batching."""
    def __init__(self, X, y):
        self.X = X[:, :, np.newaxis]     # (N, 140, 1)  – LSTM input shape
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def make_dataset(X, y, batch_size, shuffle):
    gen = ECGGenerator(X, y)
    dataset = ds.GeneratorDataset(
        gen,
        column_names=["series", "label"],
        shuffle=shuffle,
        num_parallel_workers=1,
        python_multiprocessing=False,
    )
    dataset = dataset.batch(batch_size, drop_remainder=False)
    return dataset

train_dataset = make_dataset(X_train, y_train, BATCH_SIZE, shuffle=True)
test_dataset  = make_dataset(X_test,  y_test,  BATCH_SIZE, shuffle=False)

# ── LSTM Model ─────────────────────────────────────────────────────────────────
class ECG_LSTM(nn.Cell):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,          # input: (batch, seq, feature) — same as time_major=False
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.fc      = nn.Dense(hidden_size, num_classes)

    def construct(self, x):
        # x: (batch, seq_len, input_size)
        out, (h, c) = self.lstm(x)
        # Take last time-step hidden state
        last = out[:, -1, :]
        last = self.dropout(last)
        return self.fc(last)

model = ECG_LSTM(
    input_size  = INPUT_SIZE,
    hidden_size = 128,
    num_layers  = 2,
    num_classes = NUM_CLASSES,
    dropout     = 0.3,
)

optimizer = ms_optim.Adam(
    model.trainable_params(),
    lr           = LR,
    weight_decay = 1e-4,
)
scheduler = ms_optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.CrossEntropyLoss()

total_params = sum(p.size for p in model.trainable_params())
print(f"\nModel parameters: {total_params:,}")
print("─" * 65)

# ── Fused train step (one compiled graph: forward + loss + backward + update) ──
class TrainStepCell(nn.Cell):
    def __init__(self, network, loss_fn, optimizer):
        super().__init__()
        self.network   = network
        self.loss_fn   = loss_fn
        self.optimizer = optimizer
        self.grad_fn   = mindspore.value_and_grad(
            self._forward, None, optimizer.parameters
        )

    def _forward(self, x, labels):
        logits = self.network(x)
        return self.loss_fn(logits, labels)

    def construct(self, x, labels):
        loss, grads = self.grad_fn(x, labels)
        self.optimizer(grads)
        return loss

train_cell = TrainStepCell(model, criterion, optimizer)

# ── Reusable ops ───────────────────────────────────────────────────────────────
argmax = ops.Argmax(axis=1)
equal  = ops.Equal()
cast   = ops.Cast()

# ── Training loop ──────────────────────────────────────────────────────────────
training_start = time.time()
best_val_acc   = 0.0
final_val_acc  = 0.0
final_val_loss = 0.0

for epoch in range(1, EPOCHS + 1):
    epoch_start = time.time()

    # ── Train ──────────────────────────────────────────────────────────────────
    model.set_train(True)
    train_loss_sum, correct, total, steps = 0.0, 0, 0, 0

    for x_batch, y_batch in train_dataset.create_tuple_iterator():
        loss    = train_cell(x_batch, y_batch)
        logits  = model(x_batch)                   # forward again for accuracy
        preds   = argmax(logits)
        correct += int(cast(equal(preds, y_batch), mindspore.float32).sum().asnumpy())
        total   += int(y_batch.shape[0])
        train_loss_sum += float(loss.asnumpy())
        steps   += 1

    train_acc  = correct / total
    train_loss = train_loss_sum / steps

    # ── Validation (test set) ──────────────────────────────────────────────────
    model.set_train(False)
    vc, vt, val_loss_sum, val_steps = 0, 0, 0.0, 0

    for x_batch, y_batch in test_dataset.create_tuple_iterator():
        logits       = model(x_batch)
        loss         = criterion(logits, y_batch)
        val_loss_sum += float(loss.asnumpy())
        preds         = argmax(logits)
        vc           += int(cast(equal(preds, y_batch), mindspore.float32).sum().asnumpy())
        vt           += int(y_batch.shape[0])
        val_steps    += 1

    val_acc  = vc / vt if vt else 0.0
    val_loss = val_loss_sum / val_steps if val_steps > 0 else 0.0

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        save_checkpoint(model, "ecg_lstm_best.ckpt")

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

print("\n" + "═" * 65)
print("               TRAINING COMPLETE — FINAL RESULTS")
print("═" * 65)
print(f"  Final Validation Accuracy : {final_val_acc  * 100:.2f}%")
print(f"  Best  Validation Accuracy : {best_val_acc   * 100:.2f}%")
print(f"  Final Validation Loss     : {final_val_loss:.4f}")
print(f"  Total Training Time       : {hours:02d}h {minutes:02d}m {seconds:02d}s")
print("═" * 65)

# ── Save final model ───────────────────────────────────────────────────────────
save_checkpoint(model, "ecg_lstm_final.ckpt")
print("Models saved → ecg_lstm_best.ckpt  |  ecg_lstm_final.ckpt")