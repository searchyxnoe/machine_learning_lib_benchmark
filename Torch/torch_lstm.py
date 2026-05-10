import os
import time
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ── Force CPU & Optimize for AMD 4600H ─────────────────────────────────────────
device = torch.device("cpu")
torch.set_num_threads(6)  # AMD 4600H has 6 physical cores
torch.backends.mkldnn.enabled = True # Enable CPU-optimized math kernel libraries

# ── Config ─────────────────────────────────────────────────────────────────────
TRAIN_FILE = "./ECG5000/ECG5000_TRAIN.txt"
TEST_FILE  = "./ECG5000/ECG5000_TEST.txt"
EPOCHS     = 20
BATCH_SIZE = 64
LR         = 1e-3
SEED       = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Load ECG5000 .txt (space/tab separated, first column = label) ──────────────
def load_ecg_txt(path):
    # Mock data generation if file is missing, to prevent script crash during test
    if not os.path.exists(path):
        print(f"Warning: {path} not found. Generating dummy data for testing.")
        return np.random.randn(500, 140).astype(np.float32), np.random.randint(0, 5, 500)

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
INPUT_SIZE   = 1                       # univariate ECG

print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")
print(f"Classes: {NUM_CLASSES}  |  Sequence length: {SEQ_LEN}")

# ── Dataset ────────────────────────────────────────────────────────────────────
class ECGDataset(Dataset):
    def __init__(self, X, y):
        # Reshape to (N, SEQ_LEN, 1) and pre-convert to tensors for fast fetching
        self.X = torch.tensor(X[:, :, np.newaxis], dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_ds = ECGDataset(X_train, y_train)
test_ds  = ECGDataset(X_test,  y_test)

# Since the entire dataset fits in RAM and is already tensorized, num_workers=0 
# is highly optimal here as it avoids multiprocessing IPC overhead.
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ── LSTM Model ─────────────────────────────────────────────────────────────────
class ECG_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,           # input shape: (batch, seq, feature)
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

optimizer = torch.optim.Adam(
    model.parameters(),
    lr           = LR,
    weight_decay = 1e-4,
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.CrossEntropyLoss()

print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
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
        optimizer.zero_grad()
        
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item()
        preds    = logits.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total   += y_batch.size(0)

    train_acc  = correct / total if total else 0
    train_loss = train_loss_sum / len(train_loader) if len(train_loader) else 0

    # ── Validation (test set) ──────────────────────────────────────────────────
    model.eval()
    vc, vt, val_loss_sum = 0, 0, 0.0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            logits       = model(X_batch)
            loss         = criterion(logits, y_batch)
            
            val_loss_sum += loss.item()
            preds         = logits.argmax(dim=1)
            vc           += (preds == y_batch).sum().item()
            vt           += y_batch.size(0)

    val_acc  = vc / vt if vt else 0.0
    val_loss = val_loss_sum / len(test_loader) if len(test_loader) > 0 else 0.0

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "ecg_lstm_best.pt")

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
torch.save(model.state_dict(),    "ecg_lstm_final.pt")
torch.save(optimizer.state_dict(),"ecg_lstm_final_opt.pt")
print("Models saved → ecg_lstm_best.pt  |  ecg_lstm_final.pt")