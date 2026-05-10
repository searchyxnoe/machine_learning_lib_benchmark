import os
import re
import time
import math
import random
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.io import Dataset, DataLoader
from collections import Counter

# ── Force CPU ──────────────────────────────────────────────────────────────────
paddle.set_device("cpu")

# ── Config — NeuroBERT-Tiny exact architecture ─────────────────────────────────
VOCAB_SIZE         = 30522
HIDDEN_SIZE        = 128
INTERMEDIATE_SIZE  = 512
NUM_HEADS          = 2
NUM_LAYERS         = 2
MAX_SEQ_LEN        = 128       # SMS are short, 128 is plenty
TYPE_VOCAB_SIZE    = 2
DROPOUT            = 0.1
NUM_CLASSES        = 2         # ham / spam

EPOCHS             = 20
BATCH_SIZE         = 32
LR                 = 3e-4
SEED               = 42
DATA_PATH          = "./sms_spam_collection/SMSSpamCollection"

random.seed(SEED)
np.random.seed(SEED)
paddle.seed(SEED)

# ══════════════════════════════════════════════════════════════════════════════
# 1. Simple whitespace tokenizer + vocabulary built from training data
# ══════════════════════════════════════════════════════════════════════════════
def basic_tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()

def build_vocab(texts, max_vocab=VOCAB_SIZE):
    counter = Counter()
    for t in texts:
        counter.update(basic_tokenize(t))
    # Special tokens: PAD=0, UNK=1, CLS=2, SEP=3
    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}
    for word, _ in counter.most_common(max_vocab - len(vocab)):
        vocab[word] = len(vocab)
    return vocab

def encode(text, vocab, max_len=MAX_SEQ_LEN):
    tokens = basic_tokenize(text)
    ids    = [vocab.get(t, vocab["[UNK]"]) for t in tokens]
    ids    = [vocab["[CLS]"]] + ids[:max_len - 2] + [vocab["[SEP]"]]
    pad_len = max_len - len(ids)
    mask   = [1] * len(ids) + [0] * pad_len
    ids    = ids + [vocab["[PAD]"]] * pad_len
    return ids, mask

# ══════════════════════════════════════════════════════════════════════════════
# 2. Load SMSSpamCollection (tab-separated: label \t message)
# ══════════════════════════════════════════════════════════════════════════════
print("Loading data...")
texts, labels = [], []
with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t", 1)
        if len(parts) != 2:
            continue
        label_str, text = parts
        labels.append(0 if label_str.strip() == "ham" else 1)
        texts.append(text.strip())

print(f"Total samples: {len(texts)}  |  Spam: {sum(labels)}  |  Ham: {len(labels)-sum(labels)}")

# Train/val split (80/20, stratified)
indices   = list(range(len(texts)))
random.shuffle(indices)
split     = int(0.8 * len(indices))
train_idx = indices[:split]
val_idx   = indices[split:]

train_texts  = [texts[i]  for i in train_idx]
train_labels = [labels[i] for i in train_idx]
val_texts    = [texts[i]  for i in val_idx]
val_labels   = [labels[i] for i in val_idx]

# Build vocabulary on training data only
print("Building vocabulary...")
vocab = build_vocab(train_texts)
print(f"Vocab size: {len(vocab)}")

# ══════════════════════════════════════════════════════════════════════════════
# 3. Dataset
# ══════════════════════════════════════════════════════════════════════════════
class SMSDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.data = []
        for text, label in zip(texts, labels):
            ids, mask = encode(text, vocab)
            self.data.append((
                np.array(ids,  dtype=np.int64),
                np.array(mask, dtype=np.int64),
                np.int64(label)
            ))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

train_ds = SMSDataset(train_texts, train_labels, vocab)
val_ds   = SMSDataset(val_texts,   val_labels,   vocab)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ══════════════════════════════════════════════════════════════════════════════
# 4. NeuroBERT-Tiny — built from scratch in PaddlePaddle
#    Architecture: hidden=128, intermediate=512, heads=2, layers=2
# ══════════════════════════════════════════════════════════════════════════════
class BertEmbeddings(nn.Layer):
    def __init__(self):
        super().__init__()
        self.word_embeddings     = nn.Embedding(VOCAB_SIZE,      HIDDEN_SIZE, padding_idx=0)
        self.position_embeddings = nn.Embedding(MAX_SEQ_LEN,     HIDDEN_SIZE)
        self.token_type_embeddings = nn.Embedding(TYPE_VOCAB_SIZE, HIDDEN_SIZE)
        self.layer_norm          = nn.LayerNorm(HIDDEN_SIZE, epsilon=1e-12)
        self.dropout             = nn.Dropout(DROPOUT)

    def forward(self, input_ids, token_type_ids=None):
        seq_len   = input_ids.shape[1]
        pos_ids   = paddle.arange(seq_len, dtype="int64").unsqueeze(0)
        if token_type_ids is None:
            token_type_ids = paddle.zeros_like(input_ids)
        emb = (self.word_embeddings(input_ids)
               + self.position_embeddings(pos_ids)
               + self.token_type_embeddings(token_type_ids))
        return self.dropout(self.layer_norm(emb))


class BertSelfAttention(nn.Layer):
    def __init__(self):
        super().__init__()
        self.num_heads   = NUM_HEADS
        self.head_size   = HIDDEN_SIZE // NUM_HEADS          # 64
        self.all_head    = self.num_heads * self.head_size

        self.query   = nn.Linear(HIDDEN_SIZE, self.all_head)
        self.key     = nn.Linear(HIDDEN_SIZE, self.all_head)
        self.value   = nn.Linear(HIDDEN_SIZE, self.all_head)
        self.dropout = nn.Dropout(DROPOUT)

    def transpose_for_scores(self, x):
        # (B, S, H) → (B, heads, S, head_size)
        B, S, _ = x.shape
        x = x.reshape([B, S, self.num_heads, self.head_size])
        return x.transpose([0, 2, 1, 3])

    def forward(self, hidden, attention_mask):
        Q = self.transpose_for_scores(self.query(hidden))
        K = self.transpose_for_scores(self.key(hidden))
        V = self.transpose_for_scores(self.value(hidden))

        scores = paddle.matmul(Q, K, transpose_y=True) / math.sqrt(self.head_size)
        scores = scores + attention_mask          # mask: 0 → -10000
        probs  = F.softmax(scores, axis=-1)
        probs  = self.dropout(probs)

        ctx = paddle.matmul(probs, V)            # (B, heads, S, head_size)
        ctx = ctx.transpose([0, 2, 1, 3])
        B, S, _, _ = ctx.shape
        return ctx.reshape([B, S, self.all_head])


class BertAttention(nn.Layer):
    def __init__(self):
        super().__init__()
        self.self_attn  = BertSelfAttention()
        self.out_proj   = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.layer_norm = nn.LayerNorm(HIDDEN_SIZE, epsilon=1e-12)
        self.dropout    = nn.Dropout(DROPOUT)

    def forward(self, hidden, mask):
        attn_out = self.self_attn(hidden, mask)
        attn_out = self.dropout(self.out_proj(attn_out))
        return self.layer_norm(hidden + attn_out)


class BertFFN(nn.Layer):
    def __init__(self):
        super().__init__()
        self.dense1     = nn.Linear(HIDDEN_SIZE, INTERMEDIATE_SIZE)
        self.dense2     = nn.Linear(INTERMEDIATE_SIZE, HIDDEN_SIZE)
        self.layer_norm = nn.LayerNorm(HIDDEN_SIZE, epsilon=1e-12)
        self.dropout    = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = F.gelu(self.dense1(x))
        out = self.dropout(self.dense2(out))
        return self.layer_norm(x + out)


class BertLayer(nn.Layer):
    def __init__(self):
        super().__init__()
        self.attention = BertAttention()
        self.ffn       = BertFFN()

    def forward(self, hidden, mask):
        hidden = self.attention(hidden, mask)
        return self.ffn(hidden)


class NeuroBERTTiny(nn.Layer):
    """
    Exact NeuroBERT-Tiny config trained from scratch:
      hidden=128 | intermediate=512 | layers=2 | heads=2 | vocab=30522
    """
    def __init__(self, num_classes):
        super().__init__()
        self.embeddings = BertEmbeddings()
        self.encoder    = nn.LayerList([BertLayer() for _ in range(NUM_LAYERS)])
        self.pooler     = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.dropout    = nn.Dropout(DROPOUT)
        self.classifier = nn.Linear(HIDDEN_SIZE, num_classes)

        # Xavier init for all linear layers
        for layer in self.sublayers():
            if isinstance(layer, nn.Linear):
                nn.initializer.XavierUniform()(layer.weight)
                if layer.bias is not None:
                    nn.initializer.Constant(0.0)(layer.bias)

    def forward(self, input_ids, attention_mask):
        # Build extended attention mask: (B,1,1,S) with -10000 on padding
        ext_mask = (1.0 - attention_mask.astype("float32")).unsqueeze([1, 2]) * -10000.0

        hidden = self.embeddings(input_ids)
        for layer in self.encoder:
            hidden = layer(hidden, ext_mask)

        # Pool [CLS] token
        cls_out = F.tanh(self.pooler(hidden[:, 0, :]))
        cls_out = self.dropout(cls_out)
        return self.classifier(cls_out)


model = NeuroBERTTiny(num_classes=NUM_CLASSES)
total_params = sum(p.numel().item() for p in model.parameters())
print(f"NeuroBERT-Tiny parameters: {total_params:,}")

optimizer = paddle.optimizer.AdamW(
    learning_rate = LR,
    parameters    = model.parameters(),
    weight_decay  = 0.01,
)
scheduler = paddle.optimizer.lr.CosineAnnealingDecay(LR, T_max=EPOCHS)
criterion = nn.CrossEntropyLoss()

# ══════════════════════════════════════════════════════════════════════════════
# 5. Training loop (same metric recording as previous scripts)
# ══════════════════════════════════════════════════════════════════════════════
training_start = time.time()

best_val_acc   = 0.0
final_val_acc  = 0.0
final_val_loss = 0.0

print("\n" + "─" * 75)

for epoch in range(1, EPOCHS + 1):
    epoch_start = time.time()

    # ── Train ──────────────────────────────────────────────────────────────────
    model.train()
    train_loss_sum, correct, total = 0.0, 0, 0

    for input_ids, attn_mask, labels_batch in train_loader:
        logits = model(input_ids, attn_mask)
        loss   = criterion(logits, labels_batch)
        loss.backward()
        # Gradient clipping helps small BERT models
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.clear_grad()

        train_loss_sum += float(loss)
        preds    = paddle.argmax(logits, axis=1)
        correct += int((preds == labels_batch).sum())
        total   += labels_batch.shape[0]

    train_acc  = correct / total
    train_loss = train_loss_sum / len(train_loader)

    # ── Validation ─────────────────────────────────────────────────────────────
    model.eval()
    vc, vt, val_loss_sum = 0, 0, 0.0

    with paddle.no_grad():
        for input_ids, attn_mask, labels_batch in val_loader:
            logits       = model(input_ids, attn_mask)
            loss         = criterion(logits, labels_batch)
            val_loss_sum += float(loss)
            preds         = paddle.argmax(logits, axis=1)
            vc           += int((preds == labels_batch).sum())
            vt           += labels_batch.shape[0]

    val_acc  = vc / vt if vt else 0.0
    val_loss = val_loss_sum / len(val_loader)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        paddle.save(model.state_dict(), "neurobert_tiny_best.pdparams")

    final_val_acc  = val_acc
    final_val_loss = val_loss

    epoch_time = time.time() - epoch_start
    scheduler.step()

    print(f"Epoch [{epoch:02d}/{EPOCHS}]  "
          f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}  |  "
          f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}  |  "
          f"Time: {epoch_time:.1f}s")

# ══════════════════════════════════════════════════════════════════════════════
# 6. Final summary
# ══════════════════════════════════════════════════════════════════════════════
total_time = time.time() - training_start
hours   = int(total_time // 3600)
minutes = int((total_time % 3600) // 60)
seconds = int(total_time % 60)

print("\n" + "═" * 65)
print("               TRAINING COMPLETE — FINAL RESULTS")
print("═" * 65)
print(f"  Final Validation Accuracy : {final_val_acc  * 100:.2f}%")
print(f"  Best  Validation Accuracy : {best_val_acc   * 100:.2f}%")
print(f"  Final Validation Loss     : {final_val_loss:.4f}")
print(f"  Total Training Time       : {hours:02d}h {minutes:02d}m {seconds:02d}s")
print("═" * 65)

paddle.save(model.state_dict(),     "neurobert_tiny_final.pdparams")
paddle.save(optimizer.state_dict(), "neurobert_tiny_final_opt.pdopt")
print("Saved → neurobert_tiny_best.pdparams  |  neurobert_tiny_final.pdparams")