import os
import time
import numpy as np
import paddle
import paddle.nn as nn
from paddle.io import Dataset, DataLoader

# ── Force CPU ──────────────────────────────────────────────────────────────────
paddle.set_device("cpu")

# ── Config ─────────────────────────────────────────────────────────────────────
TRAIN_FILE = "./ECG5000/ECG5000_TRAIN.txt"
TEST_FILE  = "./ECG5000/ECG5000_TEST.txt"
EPOCHS     = 20
BATCH_SIZE = 64
LR         = 1e-3
SEED       = 42

paddle.seed(SEED)
np.random.seed(SEED)

# ── Load ECG5000 .txt (space/tab separated, first column = label) ──────────────
def load_ecg_txt(path):
    data = np.loadtxt(path)          # shape: (N, 141)  → col0=label, col1..140=signal
    labels = data[:, 0].astype(np.int64)
    series = data[:, 1:].astype(np.float32)
    # ECG5000 labels are 1-indexed (1..5) → shift to 0-indexed
    labels = labels - 1
    return series, labels

print("Loading data...")
X_train, y_train = load_ecg_txt(TRAIN_FILE)
X_test,  y_test  = load_ecg_txt(TEST_FILE)

NUM_CLASSES  = len(np.unique(y_train))
SEQ_LEN      = X_train.shape[1]        # 140 time steps
INPUT_SIZE   = 1                        # univariate ECG

print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")
print(f"Classes: {NUM_CLASSES}  |  Sequence length: {SEQ_LEN}")

# ── Dataset ────────────────────────────────────────────────────────────────────
class ECGDataset(Dataset):
    def __init__(self, X, y):
        # Reshape to (N, SEQ_LEN, 1) for LSTM input
        self.X = X[:, :, np.newaxis]   # (N, 140, 1)
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_ds = ECGDataset(X_train, y_train)
test_ds  = ECGDataset(X_test,  y_test)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ── LSTM Model ─────────────────────────────────────────────────────────────────
class ECG_LSTM(nn.Layer):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            time_major  = False,          # input shape: (batch, seq, feature)
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, (h, c) = self.lstm(x)
        # Take the last time step's hidden state
        out = self.dropout(out[:, -1, :])
        return self.fc(out)

model = ECG_LSTM(
    input_size  = INPUT_SIZE,
    hidden_size = 128,
    num_layers  = 2,
    num_classes = NUM_CLASSES,
    dropout     = 0.3,
)

optimizer = paddle.optimizer.Adam(
    learning_rate = LR,
    parameters    = model.parameters(),
    weight_decay  = 1e-4,
)
scheduler = paddle.optimizer.lr.CosineAnnealingDecay(LR, T_max=EPOCHS)
criterion = nn.CrossEntropyLoss()

print(f"\nModel parameters: {sum(p.numel().item() for p in model.parameters()):,}")
print("─" * 65)

# ── Training loop ──────────────────────────────────────────────────────────────
training_start = time.time()

best_val_acc   = 0.0
final_val_acc  = 0.0
final_val_loss = 0.0

for epoch in range(1, EPOCHS + 1):
    epoch_start = time.time()

    # ── Train ──────────────────────────────────────────────────────────────────
    model.train()
    train_loss_sum, correct, total = 0.0, 0, 0

    for X_batch, y_batch in train_loader:
        X_batch = paddle.to_tensor(X_batch, dtype="float32")
        y_batch = paddle.to_tensor(y_batch, dtype="int64")

        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()

        train_loss_sum += float(loss)
        preds    = paddle.argmax(logits, axis=1)
        correct += int((preds == y_batch).sum())
        total   += y_batch.shape[0]

    train_acc  = correct / total
    train_loss = train_loss_sum / len(train_loader)

    # ── Validation (test set) ──────────────────────────────────────────────────
    model.eval()
    vc, vt, val_loss_sum = 0, 0, 0.0

    with paddle.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = paddle.to_tensor(X_batch, dtype="float32")
            y_batch = paddle.to_tensor(y_batch, dtype="int64")

            logits       = model(X_batch)
            loss         = criterion(logits, y_batch)
            val_loss_sum += float(loss)
            preds         = paddle.argmax(logits, axis=1)
            vc           += int((preds == y_batch).sum())
            vt           += y_batch.shape[0]

    val_acc  = vc / vt if vt else 0.0
    val_loss = val_loss_sum / len(test_loader)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        paddle.save(model.state_dict(), "ecg_lstm_best.pdparams")

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
paddle.save(model.state_dict(),    "ecg_lstm_final.pdparams")
paddle.save(optimizer.state_dict(),"ecg_lstm_final_opt.pdopt")
print("Models saved → ecg_lstm_best.pdparams  |  ecg_lstm_final.pdparams")