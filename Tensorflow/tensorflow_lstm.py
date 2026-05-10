import os
import time
import numpy as np

# ── Force CPU before importing TensorFlow ──────────────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

# Optimize CPU threads for parallel execution
tf.config.threading.set_inter_op_parallelism_threads(0)
tf.config.threading.set_intra_op_parallelism_threads(0)

# ── Config ─────────────────────────────────────────────────────────────────────
TRAIN_FILE = "./ECG5000/ECG5000_TRAIN.txt"
TEST_FILE  = "./ECG5000/ECG5000_TEST.txt"
EPOCHS     = 20
BATCH_SIZE = 64
LR         = 1e-3
SEED       = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)

# ── Load ECG5000 .txt (space/tab separated, first column = label) ──────────────
def load_ecg_txt(path):
    # Dummy data generation if files don't exist (for safety, you can remove this block)
    if not os.path.exists(path):
        print(f"Warning: {path} not found. Creating dummy data.")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        dummy_data = np.random.randn(500, 141)
        dummy_data[:, 0] = np.random.randint(1, 6, size=(500,))
        np.savetxt(path, dummy_data)

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

# Reshape X for LSTM input: (N, SEQ_LEN, 1)
X_train = X_train[:, :, np.newaxis]
X_test  = X_test[:, :, np.newaxis]

print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")
print(f"Classes: {NUM_CLASSES}  |  Sequence length: {SEQ_LEN}")

# ── Dataset Pipeline ───────────────────────────────────────────────────────────
AUTOTUNE = tf.data.AUTOTUNE

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_ds = train_ds.shuffle(len(X_train), seed=SEED)
train_ds = train_ds.batch(BATCH_SIZE)
train_ds = train_ds.cache().prefetch(AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test_ds = test_ds.batch(BATCH_SIZE)
test_ds = test_ds.cache().prefetch(AUTOTUNE)

# ── LSTM Model ─────────────────────────────────────────────────────────────────
class ECG_LSTM(tf.keras.Model):
    def __init__(self, hidden_size, num_layers, num_classes, dropout=0.3):
        super().__init__()
        
        # Build stacked LSTM
        self.lstm_layers = []
        for i in range(num_layers):
            return_seq = True if i < num_layers - 1 else False
            self.lstm_layers.append(
                tf.keras.layers.LSTM(
                    hidden_size, 
                    return_sequences=return_seq,
                    dropout=dropout if num_layers > 1 else 0.0
                )
            )
            
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.fc = tf.keras.layers.Dense(num_classes)

    def call(self, x, training=False):
        # x: (batch, seq_len, input_size)
        for lstm in self.lstm_layers:
            x = lstm(x, training=training)
        
        # x is now the last hidden state: (batch, hidden_size)
        out = self.dropout(x, training=training)
        return self.fc(out)

model = ECG_LSTM(
    hidden_size = 128,
    num_layers  = 2,
    num_classes = NUM_CLASSES,
    dropout     = 0.3,
)

# Dummy call to initialize weights and print parameters
model(tf.zeros([1, SEQ_LEN, INPUT_SIZE]))
print(f"\nModel parameters: {model.count_params():,}")
print("─" * 65)

# ── Initialization and Compilation ──────────────────────────────────────────────
steps_per_epoch = int(np.ceil(len(X_train) / BATCH_SIZE))
total_steps = EPOCHS * steps_per_epoch

lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=LR, 
    decay_steps=total_steps
)

optimizer = tf.keras.optimizers.Adam(
    learning_rate=lr_schedule, 
    weight_decay=1e-4
)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# ── Accelerated Graph Step Functions ───────────────────────────────────────────
@tf.function
def train_step(X_batch, y_batch):
    with tf.GradientTape() as tape:
        logits = model(X_batch, training=True)
        loss = loss_fn(y_batch, logits)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    preds = tf.argmax(logits, axis=1, output_type=tf.int64)
    correct = tf.reduce_sum(tf.cast(preds == y_batch, tf.int32))
    return loss, correct

@tf.function
def val_step(X_batch, y_batch):
    logits = model(X_batch, training=False)
    loss = loss_fn(y_batch, logits)
    
    preds = tf.argmax(logits, axis=1, output_type=tf.int64)
    correct = tf.reduce_sum(tf.cast(preds == y_batch, tf.int32))
    return loss, correct

# ── Training loop ──────────────────────────────────────────────────────────────
training_start = time.time()

best_val_acc   = 0.0
final_val_acc  = 0.0
final_val_loss = 0.0

for epoch in range(1, EPOCHS + 1):
    epoch_start = time.time()

    # ── Train ──────────────────────────────────────────────────────────────────
    train_loss_sum, correct, total, batches = 0.0, 0, 0, 0

    for X_batch, y_batch in train_ds:
        loss, corr = train_step(X_batch, y_batch)
        train_loss_sum += float(loss)
        correct        += int(corr)
        total          += int(tf.shape(y_batch)[0])
        batches        += 1

    train_acc  = correct / total if total > 0 else 0.0
    train_loss = train_loss_sum / batches if batches > 0 else 0.0

    # ── Validation (test set) ──────────────────────────────────────────────────
    vc, vt, val_loss_sum, val_batches = 0, 0, 0.0, 0

    for X_batch, y_batch in test_ds:
        loss, corr = val_step(X_batch, y_batch)
        val_loss_sum += float(loss)
        vc           += int(corr)
        vt           += int(tf.shape(y_batch)[0])
        val_batches  += 1

    val_acc  = vc / vt if vt > 0 else 0.0
    val_loss = val_loss_sum / val_batches if val_batches > 0 else 0.0

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        model.save_weights("ecg_lstm_best.weights.h5")

    final_val_acc  = val_acc
    final_val_loss = val_loss
    epoch_time = time.time() - epoch_start

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
model.save_weights("ecg_lstm_final.weights.h5")
print("Models saved → ecg_lstm_best.weights.h5  |  ecg_lstm_final.weights.h5")