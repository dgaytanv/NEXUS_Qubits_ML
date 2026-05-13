#!/usr/bin/env python3
"""
Train a proper 1D TCN for charge jump detection in qubit scans.

Architecture: True TCN with residual blocks (d=2, f=8, k=3, dil=2)
Each block contains two dilated causal convolutions with a residual skip
connection — this is what distinguishes a TCN from a plain dilated CNN.

Relation to train_jump_cnn.py:
    train_jump_cnn.py  — dilated causal CNN (no residual connections)
    train_jump_tcn.py  — proper TCN with residual blocks  ← this file

Template → Qubit mapping (from SimulatedData_Gen/Templates README):
    templ000 → Q1
    templ001 → Q2
    templ002 → Q3
    templ003 → Q4 (SC)  shield closed
    templ004 → Q4 (SO)  shield open

Usage:
    python train_jump_tcn.py --data-dir /home/export/jgaytanv/quantum/Data_Output_50k/flat
    python train_jump_tcn.py --data-dir /home/export/jgaytanv/quantum/Data_Output_50k/flat \\
        --epochs 500 --batch-size 256 --output-dir training_output_tcn_50k
"""

import os
import json
import argparse
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve
)


def set_global_determinism(seed):
    """
    Pin every source of randomness so results are fully reproducible.

    Sources controlled:
        - Python hash seed  (must be set before interpreter starts — done via
          PYTHONHASHSEED env var in the SLURM script)
        - NumPy
        - TensorFlow / Keras global seed
        - TF GPU determinism  (disables non-deterministic CUDA ops)

    Note: TF deterministic mode may be slightly slower on GPU because it
    replaces non-deterministic atomics with deterministic equivalents.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # Prevent TF from pre-allocating all GPU memory —
    # grow allocation as needed (essential for MPS shared GPU environments)
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    # Force TF to use only deterministic ops
    tf.config.experimental.enable_op_determinism()


# ── Architecture hyperparameters ──────────────────────────────────────────────
INPUT_LENGTH   = 128
DEPTH          = 2       # number of TCN residual blocks
BASE_FILTERS   = 8       # filters in block 0; doubles each block (8, 16, ...)
                         # was: 16 — halved to match original ~1,857 param budget
KERNEL_SIZE    = 3       # conv kernel size within each block
BASE_DILATION  = 2       # dilation in block d = BASE_DILATION ** d  (1, 2, 4, ...)
INPUT_CHANNELS = 2       # channel 0: raw scan, channel 1: first-difference

# ── Qubit mapping (template index → qubit label) ──────────────────────────────
QUBIT_MAP = {
    0: "Q1",
    1: "Q2",
    2: "Q3",
    3: "Q4 (SC)",
    4: "Q4 (SO)",
}


# ── TCN building blocks ───────────────────────────────────────────────────────

def residual_block(x, filters, kernel_size, dilation, block_idx):
    """
    One TCN residual block.

    Structure:
        input
          ├─ Conv1D(dil) → ReLU → BN
          │  Conv1D(dil) → ReLU → BN ──────────────────(+)── output
          └─ [1×1 Conv if channels differ] ────────────┘
    """
    residual = x

    x = layers.Conv1D(
        filters, kernel_size,
        dilation_rate=dilation,
        padding="causal",
        use_bias=True,
        kernel_initializer="he_normal",
        name=f"tcn_conv_a_{block_idx}",
    )(x)
    x = layers.ReLU(name=f"tcn_relu_a_{block_idx}")(x)
    x = layers.BatchNormalization(name=f"tcn_bn_a_{block_idx}")(x)

    x = layers.Conv1D(
        filters, kernel_size,
        dilation_rate=dilation,
        padding="causal",
        use_bias=True,
        kernel_initializer="he_normal",
        name=f"tcn_conv_b_{block_idx}",
    )(x)
    x = layers.ReLU(name=f"tcn_relu_b_{block_idx}")(x)
    x = layers.BatchNormalization(name=f"tcn_bn_b_{block_idx}")(x)

    if residual.shape[-1] != filters:
        residual = layers.Conv1D(
            filters, 1,
            use_bias=False,
            kernel_initializer="he_normal",
            name=f"tcn_res_{block_idx}",
        )(residual)

    x = layers.Add(name=f"tcn_add_{block_idx}")([x, residual])
    return x


def build_model():
    """
    Build the float32 TCN for jump detection.

    Architecture:
        Input(128, 2)
        → TCN Block 0: 2× Conv1D(8,  k=3, dil=1, causal) + residual
        → TCN Block 1: 2× Conv1D(16, k=3, dil=2, causal) + residual
        → GlobalMaxPooling1D → Dense(1) → Sigmoid
    """
    x = inp = layers.Input(shape=(INPUT_LENGTH, INPUT_CHANNELS), name="inp_signal")

    for d in range(DEPTH):
        filters  = min(BASE_FILTERS * (2 ** d), 256)
        dilation = BASE_DILATION ** d
        x = residual_block(x, filters, KERNEL_SIZE, dilation, block_idx=d)

    #x = layers.GlobalMaxPooling1D(name="global_maxpool")(x)
    #globalmaxpooling not available on hls4ml, use this instead
    x = layers.AveragePooling1D(pool_size=INPUT_LENGTH, name="avgpool")(x)
    x = layers.Flatten(name="flatten")(x)

    x = layers.Dense(1, name="dense_out")(x)
    x = layers.Activation("sigmoid", name="sigmoid")(x)

    model = keras.Model(inputs=inp, outputs=x, name="jump_tcn_c15_true")
    return model


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(data_dir):
    """
    Load scan data from flat directory of .npy files.

    Files named like:
        templ000_scan0000_jump1_j042_v0.123_js0.100.npy  → label 1
        templ000_scan0001_jump0_jNone.npy                → label 0

    Returns:
        X        : (N, 128, 2)  raw scan + diff channel
        y        : (N,)         labels (0/1)
        templids : (N,)         template index per sample (0–4)
    """
    X, y, templids = [], [], []
    all_files = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
    n_jump, n_nojump = 0, 0

    for fpath in all_files:
        fname = os.path.basename(fpath)

        if "_jump1_" in fname:
            label = 1;  n_jump += 1
        elif "_jump0_" in fname:
            label = 0;  n_nojump += 1
        else:
            continue

        try:
            templ_idx = int(fname.split("templ")[1].split("_")[0])
        except (IndexError, ValueError):
            templ_idx = -1

        data = np.load(fpath, allow_pickle=True).item()
        scan = data["Q1_scan"].astype(np.float32)

        scan = scan[:INPUT_LENGTH] if len(scan) >= INPUT_LENGTH else \
               np.pad(scan, (0, INPUT_LENGTH - len(scan)))

        diff   = np.diff(scan, prepend=scan[0]).astype(np.float32)
        sample = np.stack([scan, diff], axis=-1)

        X.append(sample)
        y.append(label)
        templids.append(templ_idx)

    print(f"  Found {n_jump} jump files and {n_nojump} no-jump files in {data_dir}")

    return (np.array(X,        dtype=np.float32),
            np.array(y,        dtype=np.float32),
            np.array(templids, dtype=np.int32))


# ── Normalization ─────────────────────────────────────────────────────────────

def normalize(X_train, X_val, X_test):
    """Per-feature standardization fitted on training set only."""
    mean = X_train.mean(axis=0, keepdims=True)
    std  = X_train.std(axis=0,  keepdims=True) + 1e-8
    return (X_train-mean)/std, (X_val-mean)/std, (X_test-mean)/std, mean, std


# ── Per-qubit efficiency ──────────────────────────────────────────────────────

def compute_qubit_efficiency(y_test, y_pred, templids_test):
    """
    Compute jump detection efficiency per qubit, matching Table S2 format.
    Efficiency = recall on jump-labelled samples for each qubit.
    Returns dict: qubit_label → (efficiency, n_correct, n_total)
    """
    results = {}
    for tidx, qname in QUBIT_MAP.items():
        mask    = (templids_test == tidx) & (y_test == 1)
        n_total = int(mask.sum())
        if n_total == 0:
            results[qname] = (float('nan'), 0, 0)
            continue
        n_correct = int((y_pred[mask] == 1).sum())
        results[qname] = (n_correct / n_total, n_correct, n_total)
    return results


def print_qubit_efficiency_table(qubit_results):
    """Print efficiency table in the style of Table S2."""
    print("\n" + "=" * 55)
    print("PER-QUBIT JUMP DETECTION EFFICIENCY  (cf. Table S2)")
    print("=" * 55)
    print(f"  {'Qubit':<12} {'Efficiency':>12}  {'Correct / Total':>18}")
    print("  " + "-" * 50)
    efficiencies = []
    for qname, (eff, n_correct, n_total) in qubit_results.items():
        if np.isnan(eff):
            print(f"  {qname:<12} {'N/A':>12}  {'no jump samples':>18}")
        else:
            efficiencies.append(eff)
            print(f"  {qname:<12} {eff:>11.4f}   {n_correct:>6d} / {n_total:<6d}")
    if efficiencies:
        print("  " + "-" * 50)
        print(f"  {'Overall':<12} {np.mean(efficiencies):>11.4f} ± {np.std(efficiencies):.4f}"
              f"  (mean ± syst.)")
    print("=" * 55)


# ── Diagnostic plots ──────────────────────────────────────────────────────────

def plot_mean_signals(X, y, output_dir):
    jump_mean   = X[y == 1].mean(axis=0)
    nojump_mean = X[y == 0].mean(axis=0)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax1.plot(nojump_mean[:, 0], label='No Jump', color='steelblue')
    ax1.plot(jump_mean[:, 0],   label='Jump',    color='crimson')
    ax1.set_ylabel('Amplitude'); ax1.set_title('Mean raw scan: jump vs no-jump')
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.plot(nojump_mean[:, 1], label='No Jump', color='steelblue')
    ax2.plot(jump_mean[:, 1],   label='Jump',    color='crimson')
    ax2.set_ylabel('Delta Amplitude (diff)'); ax2.set_xlabel('Sample index')
    ax2.set_title('Mean diff channel: jump vs no-jump')
    ax2.legend(); ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'mean_signals.png'), dpi=150)
    plt.close(fig)
    print(f"  Saved mean_signals.png")


def plot_training_history(history, output_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history["loss"],     label="Train")
    ax1.plot(history.history["val_loss"], label="Val")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Binary Cross-Entropy Loss"); ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.plot(history.history["accuracy"],     label="Train")
    ax2.plot(history.history["val_accuracy"], label="Val")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.set_title("Classification Accuracy"); ax2.legend(); ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "training_curves.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved training_curves.png")


def plot_roc_and_pr(y_true, y_prob, output_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    ax1.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    ax1.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax1.set_xlabel("False Positive Rate"); ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curve"); ax1.legend(); ax1.grid(True, alpha=0.3)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    ax2.plot(recall, precision, color="#ff7f0e", lw=2, label=f"PR (AUC = {pr_auc:.4f})")
    ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve"); ax2.legend(); ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "roc_pr_curves.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved roc_pr_curves.png")


def plot_confusion_matrix(y_true, y_pred, output_dir):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Jump", "Jump"]); ax.set_yticklabels(["No Jump", "Jump"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=16, fontweight="bold")
    fig.colorbar(im); fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved confusion_matrix.png")


# ── Results logger ────────────────────────────────────────────────────────────

def save_results_log(output_dir, args, model, history,
                     test_loss, test_acc, y_test, y_pred, y_prob,
                     qubit_results,
                     y_train, y_val, templids_train, templids_val, templids_test):
    """
    Save a human-readable results log to results.txt.
    Includes per-qubit efficiency table and per-qubit training split counts.
    """
    import datetime

    trainable    = sum(tf.size(w).numpy() for w in model.trainable_weights)
    nontrainable = sum(tf.size(w).numpy() for w in model.non_trainable_weights)
    total_params = trainable + nontrainable
    fpr, tpr, _          = roc_curve(y_test, y_prob)
    roc_auc              = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc               = auc(recall, precision)
    best_val_loss = min(history.history["val_loss"])
    best_val_acc  = max(history.history["val_accuracy"])
    n_epochs_run  = len(history.history["loss"])
    rf = sum(2 * (KERNEL_SIZE + (KERNEL_SIZE - 1) * (BASE_DILATION ** d - 1))
             for d in range(DEPTH))

    with open(os.path.join(output_dir, "results.txt"), "w") as f:
        def w(line=""):
            f.write(line + "\n")

        w("=" * 65)
        w("TRAINING RESULTS — True TCN (train_jump_tcn.py)")
        w("=" * 65)
        w(f"  Timestamp:    {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        w(f"  Output dir:   {output_dir}")
        w()
        w("── Data ──────────────────────────────────────────────────────")
        w(f"  data-dir:     {args.data_dir}")
        w(f"  Total samples (test set): {len(y_test)}")
        w(f"  No-jump (test): {int((y_test == 0).sum())}")
        w(f"  Jump (test):    {int((y_test == 1).sum())}")
        w()
        w("── Training config ───────────────────────────────────────────")
        w(f"  epochs:       {args.epochs}  (ran {n_epochs_run})")
        w(f"  batch-size:   {args.batch_size}")
        w(f"  lr:           {args.lr}")
        w(f"  seed:         {args.seed}")
        w(f"  early_stopping_patience: 40")
        w(f"  reduce_lr_factor:        0.5")
        w(f"  reduce_lr_patience:      7")
        w(f"  reduce_lr_min_lr:        1e-6")
        w(f"  deterministic_mode: enabled (tf.config.experimental.enable_op_determinism)")
        w(f"  reproduce with: python train_jump_tcn.py --seed {args.seed} --data-dir {args.data_dir} --output-dir <dir> --epochs {args.epochs} --batch-size {args.batch_size} --lr {args.lr}")
        w()
        w("── Architecture ──────────────────────────────────────────────")
        w(f"  Input:        ({INPUT_LENGTH}, {INPUT_CHANNELS})  [raw scan + diff]")
        for d in range(DEPTH):
            filters = min(BASE_FILTERS * (2 ** d), 256)
            dil     = BASE_DILATION ** d
            w(f"  TCN Block {d}: 2x Conv1D(f={filters}, k={KERNEL_SIZE}, dil={dil}, causal) + residual")
        w(f"  GlobalMaxPool -> Dense(1) -> Sigmoid")
        w(f"  Receptive field:            {rf} samples")
        w(f"  Parameters (trainable):     {trainable:,}")
        w(f"  Parameters (non-trainable): {nontrainable:,}")
        w(f"  Parameters (total):         {total_params:,}")
        w()
        w("── Validation (best epoch) ───────────────────────────────────")
        w(f"  Best val loss:     {best_val_loss:.4f}")
        w(f"  Best val accuracy: {best_val_acc:.4f}")
        w()
        w("── Test set results ──────────────────────────────────────────")
        w(f"  Test loss:         {test_loss:.4f}")
        w(f"  Test accuracy:     {test_acc:.4f}")
        w(f"  ROC AUC:           {roc_auc:.4f}")
        w(f"  PR AUC:            {pr_auc:.4f}")
        w()
        w("── Classification Report ─────────────────────────────────────")
        w(classification_report(y_test, y_pred,
                                target_names=["No Jump", "Jump"],
                                zero_division=0))
        w("── Per-Qubit Efficiency  (cf. Table S2) ──────────────────────")
        w(f"  Jump size range: 0.1e <= |dq| <= 0.5e")
        w()
        w(f"  {'Qubit':<12} {'Efficiency':>12}  {'Correct / Total':>18}")
        w("  " + "-" * 46)
        efficiencies = []
        for qname, (eff, n_correct, n_total) in qubit_results.items():
            if np.isnan(eff):
                w(f"  {qname:<12} {'N/A':>12}  {'no samples':>18}")
            else:
                efficiencies.append(eff)
                w(f"  {qname:<12} {eff:>11.4f}   {n_correct:>6d} / {n_total:<6d}")
        if efficiencies:
            w("  " + "-" * 46)
            w(f"  {'Overall':<12} {np.mean(efficiencies):>11.4f} +/- {np.std(efficiencies):.4f}"
              f"  (mean +/- syst.)")
        w()
        w("── Per-Qubit Training Split (jump samples only) ──────────────")
        w(f"  {'Qubit':<12} {'Train':>8} {'Val':>8} {'Test':>8}")
        w("  " + "-" * 40)
        for tidx, qname in QUBIT_MAP.items():
            n_tr = int(((y_train == 1) & (templids_train == tidx)).sum())
            n_va = int(((y_val   == 1) & (templids_val   == tidx)).sum())
            n_te = int(((y_test  == 1) & (templids_test  == tidx)).sum())
            w(f"  {qname:<12} {n_tr:>8} {n_va:>8} {n_te:>8}")
        w()
        w("=" * 65)

    print(f"  Saved results.txt")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train proper TCN for jump detection")
    parser.add_argument("--data-dir",   type=str, required=True)
    parser.add_argument("--epochs",     type=int,   default=500)
    parser.add_argument("--batch-size", type=int,   default=256)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--output-dir", type=str,   default="training_output_tcn")
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Reproducibility ──
    set_global_determinism(args.seed)
    print(f"  Seed {args.seed} set — deterministic mode enabled")

    # ── Save config so this run can be exactly reproduced ──
    config = {
        "script":        "train_jump_tcn.py",
        "data_dir":      args.data_dir,
        "output_dir":    args.output_dir,
        "epochs":        args.epochs,
        "batch_size":    args.batch_size,
        "lr":            args.lr,
        "seed":          args.seed,
        "early_stopping_patience": 40,
        "reduce_lr_factor":        0.5,
        "reduce_lr_patience":      7,
        "reduce_lr_min_lr":        1e-6,
        "input_length":   INPUT_LENGTH,
        "depth":          DEPTH,
        "base_filters":   BASE_FILTERS,
        "kernel_size":    KERNEL_SIZE,
        "base_dilation":  BASE_DILATION,
        "input_channels": INPUT_CHANNELS,
    }
    with open(os.path.join(args.output_dir, "config.json"), "w") as cf:
        json.dump(config, cf, indent=2)
    print(f"  Saved config.json")

    # ── Load data ──
    print("Loading data...")
    X, y, templids = load_data(args.data_dir)
    print(f"  Total: {len(X)} samples ({int(y.sum())} jump, {int(len(y)-y.sum())} no-jump)")

    print("Saving mean signal diagnostic...")
    plot_mean_signals(X, y, args.output_dir)

    # ── Split — carry templids through so test set knows qubit origin ──
    idx = np.arange(len(X))
    idx_trainval, idx_test = train_test_split(
        idx, test_size=0.15, random_state=args.seed, stratify=y
    )
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=0.176,
        random_state=args.seed, stratify=y[idx_trainval]
    )
    X_train, y_train = X[idx_train], y[idx_train]
    X_val,   y_val   = X[idx_val],   y[idx_val]
    X_test,  y_test  = X[idx_test],  y[idx_test]
    templids_train   = templids[idx_train]
    templids_val     = templids[idx_val]
    templids_test    = templids[idx_test]
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # ── Class weights ──
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    total = n_neg + n_pos
    class_weight = {0: total / (2 * n_neg), 1: total / (2 * n_pos)}
    print(f"  Class weights: no-jump={class_weight[0]:.2f}, jump={class_weight[1]:.2f}")

    # ── Normalize ──
    X_train, X_val, X_test, mean, std = normalize(X_train, X_val, X_test)
    np.save(os.path.join(args.output_dir, "norm_mean.npy"), mean)
    np.save(os.path.join(args.output_dir, "norm_std.npy"),  std)

    # ── Build & compile ──
    print("\nBuilding model...")
    model = build_model()
    model.summary()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    # ── Callbacks ──
    cb = [
        callbacks.EarlyStopping(
            monitor="val_loss", patience=40,
            restore_best_weights=True, verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=7,
            min_lr=1e-6, verbose=1
        ),
        callbacks.ModelCheckpoint(
            os.path.join(args.output_dir, "best_model.h5"),
            monitor="val_loss", save_best_only=True, verbose=1
        ),
    ]

    # ── Train ──
    print("\nTraining...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=cb,
        class_weight=class_weight,
        verbose=1,
    )

    # ── Evaluate ──
    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Test loss: {test_loss:.4f}")
    print(f"  Test accuracy: {test_acc:.4f}")

    y_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_prob > 0.5).astype(int)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=["No Jump", "Jump"],
                                zero_division=0))

    # ── Per-qubit efficiency ──
    qubit_results = compute_qubit_efficiency(y_test, y_pred, templids_test)
    print_qubit_efficiency_table(qubit_results)

    # ── Plots ──
    print("Generating plots...")
    plot_training_history(history, args.output_dir)
    plot_roc_and_pr(y_test, y_prob, args.output_dir)
    plot_confusion_matrix(y_test, y_pred, args.output_dir)

    # ── Save results log ──
    print("Saving results log...")
    save_results_log(args.output_dir, args, model, history,
                     test_loss, test_acc, y_test, y_pred, y_prob,
                     qubit_results,
                     y_train, y_val, templids_train, templids_val, templids_test)

    # ── Save model ──
    model.save(os.path.join(args.output_dir, "jump_tcn_true_float32.h5"))
    print(f"\nModel saved to {args.output_dir}/jump_tcn_true_float32.h5")

    # ── Architecture summary ──
    trainable    = sum(tf.size(w).numpy() for w in model.trainable_weights)
    nontrainable = sum(tf.size(w).numpy() for w in model.non_trainable_weights)
    rf = sum(2 * (KERNEL_SIZE + (KERNEL_SIZE - 1) * (BASE_DILATION ** d - 1))
             for d in range(DEPTH))
    print("\n" + "=" * 65)
    print("ARCHITECTURE SUMMARY (True TCN — residual blocks)")
    print("=" * 65)
    print(f"  Input:          ({INPUT_LENGTH}, {INPUT_CHANNELS})  [raw scan + first-difference]")
    print()
    for d in range(DEPTH):
        f   = min(BASE_FILTERS * (2 ** d), 256)
        dil = BASE_DILATION ** d
        print(f"  TCN Block {d}:    2x Conv1D(f={f}, k={KERNEL_SIZE}, dil={dil}, causal) + residual")
    print()
    print(f"  GlobalMaxPool -> Dense(1) -> Sigmoid")
    print(f"  Receptive field:            {rf} samples")
    print(f"  Parameters (trainable):     {trainable:,}")
    print(f"  Parameters (non-trainable): {nontrainable:,}")
    print(f"  Parameters (total):         {trainable+nontrainable:,}")
    print()
    print(f"  FPGA target: ap_fixed<16,6>, Resource strategy, io_stream")
    print(f"  Note: re-synthesize after QKeras quantization.")
    print("=" * 65)

    print("\nNEXT STEPS:")
    print("  1. Compare per-qubit efficiency with Table S2 in paper")
    print("  2. Quantize with QKeras, convert with hls4ml, synthesize on xwing")
    print("  3. Integrate with QICK firmware (see arXiv:2501.14663)")


if __name__ == "__main__":
    main()
