import os
import re
import time
import math
import random
import numpy as np
from collections import Counter

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as mnp
from mindspore import Tensor, context, set_seed
from mindspore import save_checkpoint, load_checkpoint, load_param_into_net
from mindspore.experimental import optim as ms_optim
from mindspore.common.initializer import XavierUniform, Constant
import mindspore.dataset as ds

# ── Force CPU + Static Graph ───────────────────────────────────────────────────
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

# ── Config — NeuroBERT-Tiny exact architecture ─────────────────────────────────
VOCAB_SIZE        = 30522
HIDDEN_SIZE       = 128
INTERMEDIATE_SIZE = 512
NUM_HEADS         = 2
NUM_LAYERS        = 2
MAX_SEQ_LEN       = 128
TYPE_VOCAB_SIZE   = 2
DROPOUT           = 0.1
NUM_CLASSES       = 2          # ham / spam

EPOCHS            = 20
BATCH_SIZE        = 32
LR                = 3e-4
SEED              = 42
DATA_PATH         = "./sms_spam_collection/SMSSpamCollection"

random.seed(SEED)
np.random.seed(SEED)
set_seed(SEED)

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
    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}
    for word, _ in counter.most_common(max_vocab - len(vocab)):
        vocab[word] = len(vocab)
    return vocab

def encode(text, vocab, max_len=MAX_SEQ_LEN):
    tokens  = basic_tokenize(text)
    ids     = [vocab.get(t, vocab["[UNK]"]) for t in tokens]
    ids     = [vocab["[CLS]"]] + ids[:max_len - 2] + [vocab["[SEP]"]]
    pad_len = max_len - len(ids)
    mask    = [1] * len(ids) + [0] * pad_len
    ids     = ids + [vocab["[PAD]"]] * pad_len
    return ids, mask

# ══════════════════════════════════════════════════════════════════════════════
# 2. Load SMSSpamCollection
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

indices  = list(range(len(texts)))
random.shuffle(indices)
split    = int(0.8 * len(indices))
train_idx = indices[:split]
val_idx   = indices[split:]

train_texts  = [texts[i]  for i in train_idx]
train_labels = [labels[i] for i in train_idx]
val_texts    = [texts[i]  for i in val_idx]
val_labels   = [labels[i] for i in val_idx]

print("Building vocabulary...")
vocab = build_vocab(train_texts)
print(f"Vocab size: {len(vocab)}")

# ══════════════════════════════════════════════════════════════════════════════
# 3. Pre-encode everything into numpy arrays (fast; done once, CPU only)
# ══════════════════════════════════════════════════════════════════════════════
def encode_all(texts, labels, vocab):
    all_ids, all_masks, all_labels = [], [], []
    for text, label in zip(texts, labels):
        ids, mask = encode(text, vocab)
        all_ids.append(ids)
        all_masks.append(mask)
        all_labels.append(label)
    return (np.array(all_ids,    dtype=np.int32),
            np.array(all_masks,  dtype=np.int32),
            np.array(all_labels, dtype=np.int32))

train_ids, train_masks, train_lbls = encode_all(train_texts, train_labels, vocab)
val_ids,   val_masks,   val_lbls   = encode_all(val_texts,   val_labels,   vocab)

class SMSGenerator:
    """Yields pre-encoded (input_ids, attn_mask, label) tuples."""
    def __init__(self, ids, masks, labels):
        self.ids    = ids
        self.masks  = masks
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.ids[idx], self.masks[idx], self.labels[idx]

def make_dataset(ids, masks, labels, batch_size, shuffle):
    gen = SMSGenerator(ids, masks, labels)
    dataset = ds.GeneratorDataset(
        gen,
        column_names=["input_ids", "attn_mask", "label"],
        shuffle=shuffle,
        num_parallel_workers=1,
        python_multiprocessing=False,
    )
    dataset = dataset.batch(batch_size, drop_remainder=False)
    return dataset

train_dataset = make_dataset(train_ids, train_masks, train_lbls, BATCH_SIZE, shuffle=True)
val_dataset   = make_dataset(val_ids,   val_masks,   val_lbls,   BATCH_SIZE, shuffle=False)

# ══════════════════════════════════════════════════════════════════════════════
# 4. NeuroBERT-Tiny — MindSpore implementation
#    Architecture: hidden=128 | intermediate=512 | heads=2 | layers=2
# ══════════════════════════════════════════════════════════════════════════════
class BertEmbeddings(nn.Cell):
    def __init__(self):
        super().__init__()
        self.word_embeddings       = nn.Embedding(VOCAB_SIZE,       HIDDEN_SIZE, padding_idx=0)
        self.position_embeddings   = nn.Embedding(MAX_SEQ_LEN,      HIDDEN_SIZE)
        self.token_type_embeddings = nn.Embedding(TYPE_VOCAB_SIZE,  HIDDEN_SIZE)
        self.layer_norm            = nn.LayerNorm((HIDDEN_SIZE,), epsilon=1e-12)
        self.dropout               = nn.Dropout(p=DROPOUT)
        # Fixed position ids buffer
        self.pos_ids = Tensor(np.arange(MAX_SEQ_LEN, dtype=np.int32).reshape(1, -1))

    def construct(self, input_ids, token_type_ids=None):
        seq_len = input_ids.shape[1]
        pos_ids = self.pos_ids[:, :seq_len]
        if token_type_ids is None:
            token_type_ids = ops.zeros_like(input_ids)
        emb = (self.word_embeddings(input_ids)
               + self.position_embeddings(pos_ids)
               + self.token_type_embeddings(token_type_ids))
        return self.dropout(self.layer_norm(emb))


class BertSelfAttention(nn.Cell):
    def __init__(self):
        super().__init__()
        self.num_heads = NUM_HEADS
        self.head_size = HIDDEN_SIZE // NUM_HEADS     # 64
        self.all_head  = self.num_heads * self.head_size
        self.scale     = math.sqrt(self.head_size)

        self.query   = nn.Dense(HIDDEN_SIZE, self.all_head)
        self.key     = nn.Dense(HIDDEN_SIZE, self.all_head)
        self.value   = nn.Dense(HIDDEN_SIZE, self.all_head)
        self.dropout = nn.Dropout(p=DROPOUT)
        self.softmax = nn.Softmax(axis=-1)

    def transpose_for_scores(self, x):
        # (B, S, all_head) → (B, heads, S, head_size)
        b, s, _ = x.shape
        x = x.reshape(b, s, self.num_heads, self.head_size)
        return x.transpose(0, 2, 1, 3)

    def construct(self, hidden, attention_mask):
        Q = self.transpose_for_scores(self.query(hidden))
        K = self.transpose_for_scores(self.key(hidden))
        V = self.transpose_for_scores(self.value(hidden))

        # Scaled dot-product attention
        scores = ops.BatchMatMul()(Q, K.transpose(0, 1, 3, 2)) / self.scale
        scores = scores + attention_mask      # padding positions → -10000
        probs  = self.softmax(scores)
        probs  = self.dropout(probs)

        ctx = ops.BatchMatMul()(probs, V)     # (B, heads, S, head_size)
        ctx = ctx.transpose(0, 2, 1, 3)
        b, s = ctx.shape[0], ctx.shape[1]
        return ctx.reshape(b, s, self.all_head)


class BertAttention(nn.Cell):
    def __init__(self):
        super().__init__()
        self.self_attn  = BertSelfAttention()
        self.out_proj   = nn.Dense(HIDDEN_SIZE, HIDDEN_SIZE)
        self.layer_norm = nn.LayerNorm((HIDDEN_SIZE,), epsilon=1e-12)
        self.dropout    = nn.Dropout(p=DROPOUT)

    def construct(self, hidden, mask):
        attn_out = self.self_attn(hidden, mask)
        attn_out = self.dropout(self.out_proj(attn_out))
        return self.layer_norm(hidden + attn_out)


class BertFFN(nn.Cell):
    def __init__(self):
        super().__init__()
        self.dense1     = nn.Dense(HIDDEN_SIZE, INTERMEDIATE_SIZE)
        self.dense2     = nn.Dense(INTERMEDIATE_SIZE, HIDDEN_SIZE)
        self.layer_norm = nn.LayerNorm((HIDDEN_SIZE,), epsilon=1e-12)
        self.dropout    = nn.Dropout(p=DROPOUT)
        self.gelu       = nn.GELU()

    def construct(self, x):
        out = self.gelu(self.dense1(x))
        out = self.dropout(self.dense2(out))
        return self.layer_norm(x + out)


class BertLayer(nn.Cell):
    def __init__(self):
        super().__init__()
        self.attention = BertAttention()
        self.ffn       = BertFFN()

    def construct(self, hidden, mask):
        hidden = self.attention(hidden, mask)
        return self.ffn(hidden)


class NeuroBERTTiny(nn.Cell):
    """
    NeuroBERT-Tiny trained from scratch:
      hidden=128 | intermediate=512 | layers=2 | heads=2 | vocab=30522
    """
    def __init__(self, num_classes):
        super().__init__()
        self.embeddings = BertEmbeddings()
        self.encoder    = nn.CellList([BertLayer() for _ in range(NUM_LAYERS)])
        self.pooler     = nn.Dense(HIDDEN_SIZE, HIDDEN_SIZE)
        self.dropout    = nn.Dropout(p=DROPOUT)
        self.classifier = nn.Dense(HIDDEN_SIZE, num_classes)
        self.tanh       = nn.Tanh()

        # Xavier init for all Dense layers
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(
                    mindspore.common.initializer.initializer(
                        XavierUniform(), cell.weight.shape, mindspore.float32
                    )
                )
                if cell.bias is not None:
                    cell.bias.set_data(
                        mindspore.common.initializer.initializer(
                            Constant(0.0), cell.bias.shape, mindspore.float32
                        )
                    )

    def construct(self, input_ids, attention_mask):
        # Extended mask: (B, 1, 1, S) with -10000 on padding — compiled into graph
        ext_mask = (1.0 - attention_mask.astype(mindspore.float32))
        ext_mask = ext_mask.reshape(ext_mask.shape[0], 1, 1, ext_mask.shape[1])
        ext_mask = ext_mask * -10000.0

        hidden = self.embeddings(input_ids)
        for layer in self.encoder:
            hidden = layer(hidden, ext_mask)

        # Pool [CLS] token → classifier
        cls_out = self.tanh(self.pooler(hidden[:, 0, :]))
        cls_out = self.dropout(cls_out)
        return self.classifier(cls_out)


model = NeuroBERTTiny(num_classes=NUM_CLASSES)
total_params = sum(p.size for p in model.trainable_params())
print(f"NeuroBERT-Tiny parameters: {total_params:,}")

optimizer = ms_optim.AdamW(
    model.trainable_params(),
    lr           = LR,
    weight_decay = 0.01,
)
scheduler = ms_optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.CrossEntropyLoss()

# ── Gradient clipping helper ───────────────────────────────────────────────────
clip_grad = nn.ClipByNorm()   # clips per-tensor; applied inside TrainStepCell

# ── Fused train step (one compiled graph) ──────────────────────────────────────
class TrainStepCell(nn.Cell):
    """Forward + loss + grad clip + backward + update, compiled as one graph."""
    def __init__(self, network, loss_fn, optimizer, max_norm=1.0):
        super().__init__()
        self.network    = network
        self.loss_fn    = loss_fn
        self.optimizer  = optimizer
        self.max_norm   = Tensor(max_norm, mindspore.float32)
        self.clip       = nn.ClipByNorm()
        self.grad_fn    = mindspore.value_and_grad(
            self._forward, None, optimizer.parameters
        )

    def _forward(self, input_ids, attn_mask, labels):
        logits = self.network(input_ids, attn_mask)
        return self.loss_fn(logits, labels)

    def construct(self, input_ids, attn_mask, labels):
        loss, grads = self.grad_fn(input_ids, attn_mask, labels)
        # Per-tensor L2 gradient clipping (mirrors nn.utils.clip_grad_norm_)
        grads = tuple(self.clip(g, self.max_norm) for g in grads)
        self.optimizer(grads)
        return loss

train_cell = TrainStepCell(model, criterion, optimizer, max_norm=1.0)

# ── Reusable ops ───────────────────────────────────────────────────────────────
argmax = ops.Argmax(axis=1)
equal  = ops.Equal()
cast   = ops.Cast()

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
    model.set_train(True)
    train_loss_sum, correct, total, steps = 0.0, 0, 0, 0

    for input_ids, attn_mask, labels_batch in train_dataset.create_tuple_iterator():
        loss   = train_cell(input_ids, attn_mask, labels_batch)
        logits = model(input_ids, attn_mask)           # forward for accuracy
        preds  = argmax(logits)
        correct += int(cast(equal(preds, labels_batch), mindspore.float32).sum().asnumpy())
        total   += int(labels_batch.shape[0])
        train_loss_sum += float(loss.asnumpy())
        steps  += 1

    train_acc  = correct / total
    train_loss = train_loss_sum / steps

    # ── Validation ─────────────────────────────────────────────────────────────
    model.set_train(False)
    vc, vt, val_loss_sum, val_steps = 0, 0, 0.0, 0

    for input_ids, attn_mask, labels_batch in val_dataset.create_tuple_iterator():
        logits        = model(input_ids, attn_mask)
        loss          = criterion(logits, labels_batch)
        val_loss_sum += float(loss.asnumpy())
        preds         = argmax(logits)
        vc           += int(cast(equal(preds, labels_batch), mindspore.float32).sum().asnumpy())
        vt           += int(labels_batch.shape[0])
        val_steps    += 1

    val_acc  = vc / vt if vt else 0.0
    val_loss = val_loss_sum / val_steps if val_steps > 0 else 0.0

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        save_checkpoint(model, "neurobert_tiny_best.ckpt")

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

save_checkpoint(model, "neurobert_tiny_final.ckpt")
print("Saved → neurobert_tiny_best.ckpt  |  neurobert_tiny_final.ckpt")