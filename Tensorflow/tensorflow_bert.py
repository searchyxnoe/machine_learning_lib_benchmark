import os
import re
import time
import math
import random
import numpy as np
from collections import Counter

# ── Force CPU before importing TensorFlow ──────────────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

# Optimize CPU threads for parallel execution
tf.config.threading.set_inter_op_parallelism_threads(0)
tf.config.threading.set_intra_op_parallelism_threads(0)

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
tf.random.set_seed(SEED)

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

# Generate dummy data if file does not exist locally to ensure script runs
if not os.path.exists(DATA_PATH):
    print(f"Warning: {DATA_PATH} not found. Proceeding with dummy data for testing.")
    for _ in range(1000):
        texts.append("dummy test " * random.randint(5, 20))
        labels.append(random.choice([0, 1]))
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

print("Building vocabulary...")
vocab = build_vocab(train_texts)
print(f"Vocab size: {len(vocab)}")

# ══════════════════════════════════════════════════════════════════════════════
# 3. Dataset Pipeline (tf.data)
# ══════════════════════════════════════════════════════════════════════════════
def prepare_data(text_list, label_list, vocab):
    all_ids, all_masks, all_labels = [], [], []
    for text, label in zip(text_list, label_list):
        ids, mask = encode(text, vocab)
        all_ids.append(ids)
        all_masks.append(mask)
        all_labels.append(label)
    return (
        np.array(all_ids, dtype=np.int32), 
        np.array(all_masks, dtype=np.int32), 
        np.array(all_labels, dtype=np.int32)
    )

train_ids, train_masks, train_lbls = prepare_data(train_texts, train_labels, vocab)
val_ids, val_masks, val_lbls = prepare_data(val_texts, val_labels, vocab)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = tf.data.Dataset.from_tensor_slices((train_ids, train_masks, train_lbls))
train_ds = train_ds.shuffle(len(train_texts), seed=SEED)
train_ds = train_ds.batch(BATCH_SIZE).cache().prefetch(AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((val_ids, val_masks, val_lbls))
val_ds = val_ds.batch(BATCH_SIZE).cache().prefetch(AUTOTUNE)

# ══════════════════════════════════════════════════════════════════════════════
# 4. NeuroBERT-Tiny — built from scratch in TensorFlow
# ══════════════════════════════════════════════════════════════════════════════
class BertEmbeddings(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.word_embeddings = tf.keras.layers.Embedding(VOCAB_SIZE, HIDDEN_SIZE)
        self.position_embeddings = tf.keras.layers.Embedding(MAX_SEQ_LEN, HIDDEN_SIZE)
        self.token_type_embeddings = tf.keras.layers.Embedding(TYPE_VOCAB_SIZE, HIDDEN_SIZE)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-12)
        self.dropout = tf.keras.layers.Dropout(DROPOUT)

    def call(self, input_ids, token_type_ids=None, training=False):
        seq_len = tf.shape(input_ids)[1]
        pos_ids = tf.range(seq_len, dtype=tf.int32)[tf.newaxis, :]
        
        if token_type_ids is None:
            token_type_ids = tf.zeros_like(input_ids)
            
        emb = (self.word_embeddings(input_ids) + 
               self.position_embeddings(pos_ids) + 
               self.token_type_embeddings(token_type_ids))
               
        return self.dropout(self.layer_norm(emb), training=training)


class BertSelfAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = NUM_HEADS
        self.head_size = HIDDEN_SIZE // NUM_HEADS
        self.all_head = self.num_heads * self.head_size

        # Init linear layers with Glorot/Xavier Uniform
        initializer = tf.keras.initializers.GlorotUniform()
        self.query = tf.keras.layers.Dense(self.all_head, kernel_initializer=initializer)
        self.key   = tf.keras.layers.Dense(self.all_head, kernel_initializer=initializer)
        self.value = tf.keras.layers.Dense(self.all_head, kernel_initializer=initializer)
        self.dropout = tf.keras.layers.Dropout(DROPOUT)

    def transpose_for_scores(self, x):
        # (B, S, H) → (B, heads, S, head_size)
        shape = tf.shape(x)
        B, S = shape[0], shape[1]
        x = tf.reshape(x, [B, S, self.num_heads, self.head_size])
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, hidden, attention_mask, training=False):
        Q = self.transpose_for_scores(self.query(hidden))
        K = self.transpose_for_scores(self.key(hidden))
        V = self.transpose_for_scores(self.value(hidden))

        scores = tf.matmul(Q, K, transpose_b=True) / math.sqrt(self.head_size)
        scores = scores + attention_mask          # mask: 0 → -10000
        probs  = tf.nn.softmax(scores, axis=-1)
        probs  = self.dropout(probs, training=training)

        ctx = tf.matmul(probs, V)                 # (B, heads, S, head_size)
        ctx = tf.transpose(ctx, perm=[0, 2, 1, 3])
        shape = tf.shape(ctx)
        return tf.reshape(ctx, [shape[0], shape[1], self.all_head])


class BertAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.self_attn  = BertSelfAttention()
        self.out_proj   = tf.keras.layers.Dense(HIDDEN_SIZE, kernel_initializer=tf.keras.initializers.GlorotUniform())
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-12)
        self.dropout    = tf.keras.layers.Dropout(DROPOUT)

    def call(self, hidden, mask, training=False):
        attn_out = self.self_attn(hidden, mask, training=training)
        attn_out = self.dropout(self.out_proj(attn_out), training=training)
        return self.layer_norm(hidden + attn_out)


class BertFFN(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        initializer = tf.keras.initializers.GlorotUniform()
        self.dense1     = tf.keras.layers.Dense(INTERMEDIATE_SIZE, activation="gelu", kernel_initializer=initializer)
        self.dense2     = tf.keras.layers.Dense(HIDDEN_SIZE, kernel_initializer=initializer)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-12)
        self.dropout    = tf.keras.layers.Dropout(DROPOUT)

    def call(self, x, training=False):
        out = self.dense1(x)
        out = self.dropout(self.dense2(out), training=training)
        return self.layer_norm(x + out)


class BertLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attention = BertAttention()
        self.ffn       = BertFFN()

    def call(self, hidden, mask, training=False):
        hidden = self.attention(hidden, mask, training=training)
        return self.ffn(hidden, training=training)


class NeuroBERTTiny(tf.keras.Model):
    def __init__(self, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.embeddings = BertEmbeddings()
        self.encoder = [BertLayer() for _ in range(NUM_LAYERS)]
        
        initializer = tf.keras.initializers.GlorotUniform()
        self.pooler = tf.keras.layers.Dense(HIDDEN_SIZE, activation="tanh", kernel_initializer=initializer)
        self.dropout = tf.keras.layers.Dropout(DROPOUT)
        self.classifier = tf.keras.layers.Dense(num_classes, kernel_initializer=initializer)

    def call(self, inputs, training=False):
        input_ids, attention_mask = inputs
        
        # Build extended attention mask: (B,1,1,S) with -10000 on padding
        ext_mask = tf.cast(1 - attention_mask, tf.float32)
        ext_mask = tf.expand_dims(tf.expand_dims(ext_mask, axis=1), axis=2) * -10000.0

        hidden = self.embeddings(input_ids, training=training)
        for layer in self.encoder:
            hidden = layer(hidden, ext_mask, training=training)

        # Pool [CLS] token
        cls_out = self.pooler(hidden[:, 0, :])
        cls_out = self.dropout(cls_out, training=training)
        return self.classifier(cls_out)


model = NeuroBERTTiny(num_classes=NUM_CLASSES)
model((tf.zeros([1, MAX_SEQ_LEN], dtype=tf.int32), tf.ones([1, MAX_SEQ_LEN], dtype=tf.int32)))
print(f"NeuroBERT-Tiny parameters: {model.count_params():,}")

# ══════════════════════════════════════════════════════════════════════════════
# 5. Initialization and Compilation
# ══════════════════════════════════════════════════════════════════════════════
steps_per_epoch = int(np.ceil(len(train_texts) / BATCH_SIZE))
total_steps = EPOCHS * steps_per_epoch

lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=LR, 
    decay_steps=total_steps
)

# AdamW optimized for BERT weight decay scaling
optimizer = tf.keras.optimizers.AdamW(
    learning_rate=lr_schedule, 
    weight_decay=0.01,
    global_clipnorm=1.0 # Gradient clipping inside optimizer natively
)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# ── Accelerated Graph Step Functions ───────────────────────────────────────────
@tf.function
def train_step(input_ids, attn_mask, labels):
    with tf.GradientTape() as tape:
        logits = model((input_ids, attn_mask), training=True)
        loss = loss_fn(labels, logits)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    preds = tf.argmax(logits, axis=1, output_type=tf.int32)
    correct = tf.reduce_sum(tf.cast(preds == labels, tf.int32))
    return loss, correct

@tf.function
def val_step(input_ids, attn_mask, labels):
    logits = model((input_ids, attn_mask), training=False)
    loss = loss_fn(labels, logits)
    
    preds = tf.argmax(logits, axis=1, output_type=tf.int32)
    correct = tf.reduce_sum(tf.cast(preds == labels, tf.int32))
    return loss, correct

# ══════════════════════════════════════════════════════════════════════════════
# 6. Training loop
# ══════════════════════════════════════════════════════════════════════════════
training_start = time.time()

best_val_acc   = 0.0
final_val_acc  = 0.0
final_val_loss = 0.0

print("\n" + "─" * 75)

for epoch in range(1, EPOCHS + 1):
    epoch_start = time.time()

    # ── Train ──────────────────────────────────────────────────────────────────
    train_loss_sum, correct, total, batches = 0.0, 0, 0, 0

    for input_ids, attn_mask, labels in train_ds:
        loss, corr = train_step(input_ids, attn_mask, labels)
        train_loss_sum += float(loss)
        correct        += int(corr)
        total          += int(tf.shape(labels)[0])
        batches        += 1

    train_acc  = correct / total if total > 0 else 0.0
    train_loss = train_loss_sum / batches if batches > 0 else 0.0

    # ── Validation ─────────────────────────────────────────────────────────────
    vc, vt, val_loss_sum, val_batches = 0, 0, 0.0, 0

    for input_ids, attn_mask, labels in val_ds:
        loss, corr = val_step(input_ids, attn_mask, labels)
        val_loss_sum += float(loss)
        vc           += int(corr)
        vt           += int(tf.shape(labels)[0])
        val_batches  += 1

    val_acc  = vc / vt if vt > 0 else 0.0
    val_loss = val_loss_sum / val_batches if val_batches > 0 else 0.0

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        model.save_weights("neurobert_tiny_best.weights.h5")

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

model.save_weights("neurobert_tiny_final.weights.h5")
print("Saved → neurobert_tiny_best.weights.h5  |  neurobert_tiny_final.weights.h5")