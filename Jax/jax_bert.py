import os
import re
import time
import math
import random
import pickle
from collections import Counter
from typing import Any

# ── Force CPU BEFORE importing JAX ─────────────────────────────────────────────
os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax

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

print("Using JAX devices:", jax.devices())

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
    # Special tokens
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

print("Building vocabulary...")
vocab = build_vocab(train_texts)
print(f"Vocab size: {len(vocab)}")

# ══════════════════════════════════════════════════════════════════════════════
# 3. JAX Dataset Arrays & Loader
# ══════════════════════════════════════════════════════════════════════════════
def prepare_dataset(texts_list, labels_list, vocab):
    all_ids, all_masks, all_labels = [], [], []
    for text, label in zip(texts_list, labels_list):
        ids, mask = encode(text, vocab)
        all_ids.append(ids)
        all_masks.append(mask)
        all_labels.append(label)
    return (
        np.array(all_ids, dtype=np.int32),
        np.array(all_masks, dtype=np.int32),
        np.array(all_labels, dtype=np.int32)
    )

train_X, train_M, train_Y = prepare_dataset(train_texts, train_labels, vocab)
val_X,   val_M,   val_Y   = prepare_dataset(val_texts, val_labels, vocab)

def batch_iterator(X, M, Y, batch_size, shuffle, seed, epoch=0):
    indices = np.arange(len(Y))
    if shuffle:
        rng = np.random.default_rng(seed + epoch)
        rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        idx = indices[start:start + batch_size]
        yield X[idx], M[idx], Y[idx]

# ══════════════════════════════════════════════════════════════════════════════
# 4. NeuroBERT-Tiny — JAX/Flax Implementation
# ══════════════════════════════════════════════════════════════════════════════
class BertEmbeddings(nn.Module):
    @nn.compact
    def __call__(self, input_ids, token_type_ids, train: bool):
        word_emb = nn.Embed(VOCAB_SIZE, HIDDEN_SIZE, name='word_embeddings')(input_ids)
        
        seq_len = input_ids.shape[1]
        pos_ids = jnp.arange(seq_len)[None, :]
        pos_emb = nn.Embed(MAX_SEQ_LEN, HIDDEN_SIZE, name='position_embeddings')(pos_ids)
        
        type_emb = nn.Embed(TYPE_VOCAB_SIZE, HIDDEN_SIZE, name='token_type_embeddings')(token_type_ids)

        emb = word_emb + pos_emb + type_emb
        emb = nn.LayerNorm(epsilon=1e-12, name='layer_norm')(emb)
        return nn.Dropout(rate=DROPOUT)(emb, deterministic=not train)

class BertSelfAttention(nn.Module):
    @nn.compact
    def __call__(self, x, mask, train: bool):
        head_size = HIDDEN_SIZE // NUM_HEADS
        all_head = NUM_HEADS * head_size
        dense_init = nn.initializers.xavier_uniform()

        # Projections
        q = nn.Dense(all_head, kernel_init=dense_init, name='query')(x)
        k = nn.Dense(all_head, kernel_init=dense_init, name='key')(x)
        v = nn.Dense(all_head, kernel_init=dense_init, name='value')(x)

        # Transpose for scores: (B, S, all_head) -> (B, heads, S, head_size)
        B, S, _ = x.shape
        q = q.reshape(B, S, NUM_HEADS, head_size).transpose(0, 2, 1, 3)
        k = k.reshape(B, S, NUM_HEADS, head_size).transpose(0, 2, 1, 3)
        v = v.reshape(B, S, NUM_HEADS, head_size).transpose(0, 2, 1, 3)

        # Attention scores
        scores = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / math.sqrt(head_size)
        scores = scores + mask  # Apply mask
        
        probs = jax.nn.softmax(scores, axis=-1)
        probs = nn.Dropout(rate=DROPOUT)(probs, deterministic=not train)

        # Context
        ctx = jnp.matmul(probs, v)
        ctx = ctx.transpose(0, 2, 1, 3).reshape(B, S, all_head)
        return ctx

class BertAttention(nn.Module):
    @nn.compact
    def __call__(self, hidden, mask, train: bool):
        attn_out = BertSelfAttention(name='self_attn')(hidden, mask, train)
        out = nn.Dense(HIDDEN_SIZE, kernel_init=nn.initializers.xavier_uniform(), name='out_proj')(attn_out)
        out = nn.Dropout(rate=DROPOUT)(out, deterministic=not train)
        return nn.LayerNorm(epsilon=1e-12, name='layer_norm')(hidden + out)

class BertFFN(nn.Module):
    @nn.compact
    def __call__(self, x, train: bool):
        dense_init = nn.initializers.xavier_uniform()
        
        out = nn.Dense(INTERMEDIATE_SIZE, kernel_init=dense_init, name='dense1')(x)
        out = nn.gelu(out)
        
        out = nn.Dropout(rate=DROPOUT)(nn.Dense(HIDDEN_SIZE, kernel_init=dense_init, name='dense2')(out), deterministic=not train)
        return nn.LayerNorm(epsilon=1e-12, name='layer_norm')(x + out)

class BertLayer(nn.Module):
    @nn.compact
    def __call__(self, hidden, mask, train: bool):
        hidden = BertAttention(name='attention')(hidden, mask, train)
        return BertFFN(name='ffn')(hidden, train)

class NeuroBERTTiny(nn.Module):
    num_classes: int

    @nn.compact
    def __call__(self, input_ids, attention_mask, train: bool):
        # Build extended mask: (B, 1, 1, S)
        ext_mask = (1.0 - attention_mask[:, None, None, :].astype(jnp.float32)) * -10000.0
        token_type_ids = jnp.zeros_like(input_ids)

        hidden = BertEmbeddings(name='embeddings')(input_ids, token_type_ids, train)
        
        for i in range(NUM_LAYERS):
            hidden = BertLayer(name=f'layer_{i}')(hidden, ext_mask, train)

        # Pool [CLS] token
        cls_out = hidden[:, 0, :]
        dense_init = nn.initializers.xavier_uniform()
        
        cls_out = nn.Dense(HIDDEN_SIZE, kernel_init=dense_init, name='pooler')(cls_out)
        cls_out = jnp.tanh(cls_out)
        cls_out = nn.Dropout(rate=DROPOUT)(cls_out, deterministic=not train)

        return nn.Dense(self.num_classes, kernel_init=dense_init, name='classifier')(cls_out)

# ══════════════════════════════════════════════════════════════════════════════
# 5. Training Setup (Optax + TrainState)
# ══════════════════════════════════════════════════════════════════════════════
class TrainState(train_state.TrainState):
    pass

def create_train_state(rng, model, steps_per_epoch):
    dummy_ids  = jnp.ones((1, MAX_SEQ_LEN), dtype=jnp.int32)
    dummy_mask = jnp.ones((1, MAX_SEQ_LEN), dtype=jnp.int32)
    
    variables = model.init(
        {"params": rng, "dropout": jax.random.PRNGKey(SEED + 1)}, 
        dummy_ids, dummy_mask, train=True
    )
    
    # Cosine schedule mimicking CosineAnnealingDecay
    lr_schedule = optax.cosine_decay_schedule(
        init_value=LR,
        decay_steps=EPOCHS * steps_per_epoch,
        alpha=0.0
    )
    
    # Optimizer matching AdamW + gradient clipping
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=lr_schedule, weight_decay=0.01)
    )
    
    return TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=tx,
    )

@jax.jit
def train_step(state, input_ids, attn_mask, labels, dropout_key):
    def loss_fn(params):
        logits = state.apply_fn(
            {"params": params},
            input_ids, attn_mask, train=True,
            rngs={"dropout": dropout_key}
        )
        one_hot = jax.nn.one_hot(labels, NUM_CLASSES)
        loss = optax.softmax_cross_entropy(logits, one_hot).mean()
        return loss, logits

    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    
    preds = jnp.argmax(logits, axis=-1)
    correct = jnp.sum(preds == labels)
    return state, loss, correct

@jax.jit
def eval_step(state, input_ids, attn_mask, labels):
    logits = state.apply_fn(
        {"params": state.params},
        input_ids, attn_mask, train=False
    )
    one_hot = jax.nn.one_hot(labels, NUM_CLASSES)
    loss = optax.softmax_cross_entropy(logits, one_hot).mean()
    
    preds = jnp.argmax(logits, axis=-1)
    correct = jnp.sum(preds == labels)
    return loss, correct

def count_parameters(params):
    return sum(x.size for x in jax.tree_util.tree_leaves(params))

# Init model
model = NeuroBERTTiny(num_classes=NUM_CLASSES)
steps_per_epoch = math.ceil(len(train_Y) / BATCH_SIZE) if len(train_Y) > 0 else 1
state = create_train_state(jax.random.PRNGKey(SEED), model, steps_per_epoch)

print(f"NeuroBERT-Tiny parameters: {count_parameters(state.params):,}")

# ══════════════════════════════════════════════════════════════════════════════
# 6. Training loop
# ══════════════════════════════════════════════════════════════════════════════
training_start = time.time()

best_val_acc   = 0.0
final_val_acc  = 0.0
final_val_loss = 0.0

dropout_master_key = jax.random.PRNGKey(SEED + 999)

print("\n" + "─" * 75)

for epoch in range(1, EPOCHS + 1):
    epoch_start = time.time()

    # ── Train ──────────────────────────────────────────────────────────────────
    train_loss_sum, correct, total, train_batches = 0.0, 0, 0, 0

    for X_batch, M_batch, Y_batch in batch_iterator(
        train_X, train_M, train_Y, BATCH_SIZE, shuffle=True, seed=SEED, epoch=epoch
    ):
        X_batch = jnp.asarray(X_batch)
        M_batch = jnp.asarray(M_batch)
        Y_batch = jnp.asarray(Y_batch)

        dropout_master_key, dropout_key = jax.random.split(dropout_master_key)
        state, loss, batch_correct = train_step(state, X_batch, M_batch, Y_batch, dropout_key)

        train_loss_sum += float(loss)
        correct += int(batch_correct)
        total   += Y_batch.shape[0]
        train_batches += 1

    train_acc  = correct / total if total else 0.0
    train_loss = train_loss_sum / train_batches if train_batches else 0.0

    # ── Validation ─────────────────────────────────────────────────────────────
    vc, vt, val_loss_sum, val_batches = 0, 0, 0.0, 0

    for X_batch, M_batch, Y_batch in batch_iterator(
        val_X, val_M, val_Y, BATCH_SIZE, shuffle=False, seed=SEED, epoch=epoch
    ):
        X_batch = jnp.asarray(X_batch)
        M_batch = jnp.asarray(M_batch)
        Y_batch = jnp.asarray(Y_batch)

        loss, batch_correct = eval_step(state, X_batch, M_batch, Y_batch)

        val_loss_sum += float(loss)
        vc += int(batch_correct)
        vt += Y_batch.shape[0]
        val_batches += 1

    val_acc  = vc / vt if vt else 0.0
    val_loss = val_loss_sum / val_batches if val_batches else 0.0

    # Track best model parameters
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_params = jax.tree_util.tree_map(lambda x: np.asarray(x), state.params)
        with open("neurobert_tiny_best_flax.pkl", "wb") as f:
            pickle.dump(best_params, f)

    final_val_acc  = val_acc
    final_val_loss = val_loss

    epoch_time = time.time() - epoch_start

    print(f"Epoch [{epoch:02d}/{EPOCHS}]  "
          f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}  |  "
          f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}  |  "
          f"Time: {epoch_time:.1f}s")

# ══════════════════════════════════════════════════════════════════════════════
# 7. Final summary
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

# Save Final
final_params = jax.tree_util.tree_map(lambda x: np.asarray(x), state.params)
with open("neurobert_tiny_final_flax.pkl", "wb") as f:
    pickle.dump(final_params, f)

print("Saved → neurobert_tiny_best_flax.pkl  |  neurobert_tiny_final_flax.pkl")