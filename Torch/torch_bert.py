import os
import re
import time
import math
import random
import numpy as np
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ── Force CPU & Optimize for AMD 4600H ─────────────────────────────────────────
device = torch.device("cpu")
torch.set_num_threads(6)  # AMD 4600H has 6 physical cores
torch.backends.mkldnn.enabled = True # Enable CPU-optimized math kernel libraries

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
torch.manual_seed(SEED)

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

# Fallback for testing if file doesn't exist to prevent crash
if not os.path.exists(DATA_PATH):
    print(f"Warning: {DATA_PATH} not found. Generating dummy data for testing.")
    texts  = ["hello world how are you"] * 400 + ["win a free prize now claim"] * 100
    labels = [0] * 400 + [1] * 100
else:
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
        self.input_ids = []
        self.masks     = []
        # Pre-convert labels to tensor
        self.labels    = torch.tensor(labels, dtype=torch.long)
        
        for text in texts:
            ids, mask = encode(text, vocab)
            self.input_ids.append(ids)
            self.masks.append(mask)
            
        # Pre-convert lists to unified tensors to prevent runtime allocation
        self.input_ids = torch.tensor(self.input_ids, dtype=torch.long)
        self.masks     = torch.tensor(self.masks, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.masks[idx], self.labels[idx]

train_ds = SMSDataset(train_texts, train_labels, vocab)
val_ds   = SMSDataset(val_texts,   val_labels,   vocab)

# num_workers=0 is optimal here because the entire tensorized dataset fits in RAM
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ══════════════════════════════════════════════════════════════════════════════
# 4. NeuroBERT-Tiny — built from scratch in PyTorch
#    Architecture: hidden=128, intermediate=512, heads=2, layers=2
# ══════════════════════════════════════════════════════════════════════════════
class BertEmbeddings(nn.Module):
    def __init__(self):
        super().__init__()
        self.word_embeddings       = nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE, padding_idx=0)
        self.position_embeddings   = nn.Embedding(MAX_SEQ_LEN, HIDDEN_SIZE)
        self.token_type_embeddings = nn.Embedding(TYPE_VOCAB_SIZE, HIDDEN_SIZE)
        self.layer_norm            = nn.LayerNorm(HIDDEN_SIZE, eps=1e-12)
        self.dropout               = nn.Dropout(DROPOUT)

    def forward(self, input_ids, token_type_ids=None):
        seq_len = input_ids.size(1)
        pos_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
            
        emb = (self.word_embeddings(input_ids)
               + self.position_embeddings(pos_ids)
               + self.token_type_embeddings(token_type_ids))
        
        return self.dropout(self.layer_norm(emb))


class BertSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads   = NUM_HEADS
        self.head_size   = HIDDEN_SIZE // NUM_HEADS
        self.all_head    = self.num_heads * self.head_size

        self.query   = nn.Linear(HIDDEN_SIZE, self.all_head)
        self.key     = nn.Linear(HIDDEN_SIZE, self.all_head)
        self.value   = nn.Linear(HIDDEN_SIZE, self.all_head)
        self.dropout = nn.Dropout(DROPOUT)

    def transpose_for_scores(self, x):
        # (B, S, H) → (B, heads, S, head_size)
        B, S, _ = x.size()
        x = x.view(B, S, self.num_heads, self.head_size)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden, attention_mask):
        Q = self.transpose_for_scores(self.query(hidden))
        K = self.transpose_for_scores(self.key(hidden))
        V = self.transpose_for_scores(self.value(hidden))

        # PyTorch uses .transpose() for swapping exactly two dimensions
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.head_size)
        scores = scores + attention_mask          # mask: 0 → -10000
        probs  = F.softmax(scores, dim=-1)
        probs  = self.dropout(probs)

        ctx = torch.matmul(probs, V)              # (B, heads, S, head_size)
        # .contiguous() ensures cache-locality for CPU reshaping
        ctx = ctx.permute(0, 2, 1, 3).contiguous()
        B, S, _, _ = ctx.size()
        return ctx.view(B, S, self.all_head)


class BertAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn  = BertSelfAttention()
        self.out_proj   = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.layer_norm = nn.LayerNorm(HIDDEN_SIZE, eps=1e-12)
        self.dropout    = nn.Dropout(DROPOUT)

    def forward(self, hidden, mask):
        attn_out = self.self_attn(hidden, mask)
        attn_out = self.dropout(self.out_proj(attn_out))
        return self.layer_norm(hidden + attn_out)


class BertFFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense1     = nn.Linear(HIDDEN_SIZE, INTERMEDIATE_SIZE)
        self.dense2     = nn.Linear(INTERMEDIATE_SIZE, HIDDEN_SIZE)
        self.layer_norm = nn.LayerNorm(HIDDEN_SIZE, eps=1e-12)
        self.dropout    = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = F.gelu(self.dense1(x))
        out = self.dropout(self.dense2(out))
        return self.layer_norm(x + out)


class BertLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = BertAttention()
        self.ffn       = BertFFN()

    def forward(self, hidden, mask):
        hidden = self.attention(hidden, mask)
        return self.ffn(hidden)


class NeuroBERTTiny(nn.Module):
    """
    Exact NeuroBERT-Tiny config trained from scratch:
      hidden=128 | intermediate=512 | layers=2 | heads=2 | vocab=30522
    """
    def __init__(self, num_classes):
        super().__init__()
        self.embeddings = BertEmbeddings()
        self.encoder    = nn.ModuleList([BertLayer() for _ in range(NUM_LAYERS)])
        self.pooler     = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.dropout    = nn.Dropout(DROPOUT)
        self.classifier = nn.Linear(HIDDEN_SIZE, num_classes)

        # Xavier init for all linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, input_ids, attention_mask):
        # Build extended attention mask: (B,1,1,S) with -10000 on padding
        ext_mask = (1.0 - attention_mask.float()).unsqueeze(1).unsqueeze(2) * -10000.0

        hidden = self.embeddings(input_ids)
        for layer in self.encoder:
            hidden = layer(hidden, ext_mask)

        # Pool [CLS] token
        cls_out = torch.tanh(self.pooler(hidden[:, 0, :]))
        cls_out = self.dropout(cls_out)
        return self.classifier(cls_out)


model = NeuroBERTTiny(num_classes=NUM_CLASSES)
total_params = sum(p.numel() for p in model.parameters())
print(f"NeuroBERT-Tiny parameters: {total_params:,}")

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr           = LR,
    weight_decay = 0.01,
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.CrossEntropyLoss()

# ══════════════════════════════════════════════════════════════════════════════
# 5. Training loop
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
        optimizer.zero_grad()
        
        logits = model(input_ids, attn_mask)
        loss   = criterion(logits, labels_batch)
        loss.backward()
        
        # Gradient clipping helps small BERT models stabilize
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss_sum += loss.item()
        preds    = logits.argmax(dim=1)
        correct += (preds == labels_batch).sum().item()
        total   += labels_batch.size(0)

    train_acc  = correct / total if total else 0
    train_loss = train_loss_sum / len(train_loader) if len(train_loader) else 0

    # ── Validation ─────────────────────────────────────────────────────────────
    model.eval()
    vc, vt, val_loss_sum = 0, 0, 0.0

    with torch.no_grad():
        for input_ids, attn_mask, labels_batch in val_loader:
            logits       = model(input_ids, attn_mask)
            loss         = criterion(logits, labels_batch)
            
            val_loss_sum += loss.item()
            preds         = logits.argmax(dim=1)
            vc           += (preds == labels_batch).sum().item()
            vt           += labels_batch.size(0)

    val_acc  = vc / vt if vt else 0.0
    val_loss = val_loss_sum / len(val_loader) if len(val_loader) else 0.0

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "neurobert_tiny_best.pt")

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

torch.save(model.state_dict(),     "neurobert_tiny_final.pt")
torch.save(optimizer.state_dict(), "neurobert_tiny_final_opt.pt")
print("Saved → neurobert_tiny_best.pt  |  neurobert_tiny_final.pt")