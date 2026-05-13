#!/usr/bin/env python3
"""
Train a 4-layer dilated CNN matching the TCN architecture but WITHOUT residual connections.

Purpose: Ablation study to determine whether the TCN's advantage over the
2-layer dilated CNN comes from the residual connections or from the deeper
architecture (4 conv layers instead of 2).

Architecture comparison:
    train_jump_cnn.py      — 2 conv layers (16, 32 filters), no residual
    train_jump_cnn_deep.py — 4 conv layers (8, 8, 16, 16 filters), no residual  ← this file
    train_jump_tcn.py      — 4 conv layers (8, 8, 16, 16 filters), WITH residual

If this model matches the TCN, the residual connections don't matter.
If this model matches the shallow CNN, depth alone isn't enough.
If this model falls between them, both depth and residuals contribute.

Usage:
    python train_jump_cnn_deep.py --data-dir /home/export/jgaytanv/quantum/Data_Output_100k/flat
    python train_jump_cnn_deep.py --data-dir /home/export/jgaytanv/quantum/Data_Output_100k/flat \
        --output-dir training_output_cnn_deep_100k
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
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.enable_op_determinism()


# ── Architecture (matches TCN layer-for-layer, minus residual connections) ──
INPUT_LENGTH   = 128
INPUT_CHANNELS = 2
# Same conv layers as TCN blocks:
#   Block 0: Conv(8, dil=1), Conv(8, dil=1)
#   Block 1: Conv(16, dil=2), Conv(16, dil=2)
CONV_CONFIGS = [
    {"filters": 8,  "dilation": 1},   # was tcn_conv_a_0
    {"filters": 8,  "dilation": 1},   # was tcn_conv_b_0
    {"filters": 16, "dilation": 2},   # was tcn_conv_a_1
    {"filters": 16, "dilation": 2},   # was tcn_conv_b_1
]
KERNEL_SIZE = 3

QUBIT_MAP = {0: "Q1", 1: "Q2", 2: "Q3", 3: "Q4 (SC)", 4: "Q4 (SO)"}


def build_model():
    """
    Build a 4-layer dilated causal CNN (no residual connections).

    Architecture:
        Input(128, 2)
        → Conv1D(8,  k=3, dil=1, causal) → ReLU → BN
        → Conv1D(8,  k=3, dil=1, causal) → ReLU → BN
        → Conv1D(16, k=3, dil=2, causal) → ReLU → BN
        → Conv1D(16, k=3, dil=2, causal) → ReLU → BN
        → GlobalMaxPooling1D → Dense(1) → Sigmoid

    This is the same sequence of convolutions as the TCN, but without
    the residual skip connections or 1x1 projection convolutions.
    """
    x = inp = layers.Input(shape=(INPUT_LENGTH, INPUT_CHANNELS), name="inp_signal")

    for i, cfg in enumerate(CONV_CONFIGS):
        x = layers.Conv1D(
            cfg["filters"], KERNEL_SIZE,
            dilation_rate=cfg["dilation"],
            padding="causal",
            use_bias=True,
            kernel_initializer="he_normal",
            name=f"conv1d_{i}",
        )(x)
        x = layers.ReLU(name=f"relu_{i}")(x)
        x = layers.BatchNormalization(name=f"bn_{i}")(x)

    x = layers.GlobalMaxPooling1D(name="global_maxpool")(x)
    x = layers.Dense(1, name="dense_out")(x)
    x = layers.Activation("sigmoid", name="sigmoid")(x)

    model = keras.Model(inputs=inp, outputs=x, name="jump_cnn_deep")
    return model


# ── Data loading ──

def load_data(data_dir):
    X, y, templids = [], [], []
    all_files = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
    n_jump, n_nojump = 0, 0
    for fpath in all_files:
        fname = os.path.basename(fpath)
        if "_jump1_" in fname:
            label = 1; n_jump += 1
        elif "_jump0_" in fname:
            label = 0; n_nojump += 1
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
    print(f"  Found {n_jump} jump files and {n_nojump} no-jump files")
    return (np.array(X, dtype=np.float32),
            np.array(y, dtype=np.float32),
            np.array(templids, dtype=np.int32))


def normalize(X_train, X_val, X_test):
    mean = X_train.mean(axis=0, keepdims=True)
    std  = X_train.std(axis=0, keepdims=True) + 1e-8
    return (X_train-mean)/std, (X_val-mean)/std, (X_test-mean)/std, mean, std


def compute_qubit_efficiency(y_test, y_pred, templids_test):
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
        print(f"  {'Overall':<12} {np.mean(efficiencies):>11.4f} +/- {np.std(efficiencies):.4f}")
    print("=" * 55)


# ── Plots ──

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

def plot_roc_and_pr(y_true, y_prob, output_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    ax1.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    ax1.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax1.set_xlabel("FPR"); ax1.set_ylabel("TPR")
    ax1.set_title("ROC Curve"); ax1.legend(); ax1.grid(True, alpha=0.3)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    ax2.plot(recall, precision, color="#ff7f0e", lw=2, label=f"PR (AUC = {pr_auc:.4f})")
    ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
    ax2.set_title("PR Curve"); ax2.legend(); ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "roc_pr_curves.png"), dpi=150)
    plt.close(fig)

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


# ── Results logger ──

def save_results_log(output_dir, args, model, history,
                     test_loss, test_acc, y_test, y_pred, y_prob,
                     qubit_results,
                     y_train, y_val, templids_train, templids_val, templids_test):
    import datetime

    trainable    = sum(tf.size(w).numpy() for w in model.trainable_weights)
    nontrainable = sum(tf.size(w).numpy() for w in model.non_trainable_weights)
    total_params = trainable + nontrainable
    fpr, tpr, _  = roc_curve(y_test, y_prob)
    roc_auc      = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    pr_auc       = auc(rec, prec)
    best_val_loss = min(history.history["val_loss"])
    best_val_acc  = max(history.history["val_accuracy"])
    n_epochs_run  = len(history.history["loss"])

    with open(os.path.join(output_dir, "results.txt"), "w") as f:
        def w(line=""):
            f.write(line + "\n")

        w("=" * 65)
        w("TRAINING RESULTS — Deep Dilated CNN (train_jump_cnn_deep.py)")
        w("=" * 65)
        w(f"  Timestamp:    {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        w(f"  Output dir:   {output_dir}")
        w()
        w("── Purpose ───────────────────────────────────────────────────")
        w(f"  Ablation: same conv layers as TCN, no residual connections")
        w(f"  Tests whether TCN advantage comes from residuals or depth")
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
        w()
        w("── Architecture ──────────────────────────────────────────────")
        w(f"  Input:        ({INPUT_LENGTH}, {INPUT_CHANNELS})  [raw scan + diff]")
        for i, cfg in enumerate(CONV_CONFIGS):
            w(f"  Conv1D_{i}:    filters={cfg['filters']}, kernel={KERNEL_SIZE}, dilation={cfg['dilation']}, causal")
            w(f"  ReLU_{i} -> BatchNorm_{i}")
        w(f"  GlobalMaxPool -> Dense(1) -> Sigmoid")
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
                                target_names=["No Jump", "Jump"], zero_division=0))
        w("── Per-Qubit Efficiency ──────────────────────────────────────")
        w(f"  {'Qubit':<12} {'Efficiency':>12}  {'Correct / Total':>18}")
        w("  " + "-" * 46)
        effs = []
        for qname, (eff, nc, nt) in qubit_results.items():
            if np.isnan(eff):
                w(f"  {qname:<12} {'N/A':>12}  {'no samples':>18}")
            else:
                effs.append(eff)
                w(f"  {qname:<12} {eff:>11.4f}   {nc:>6d} / {nt:<6d}")
        if effs:
            w("  " + "-" * 46)
            w(f"  {'Overall':<12} {np.mean(effs):>11.4f} +/- {np.std(effs):.4f}")
        w()
        w("── Comparison ───────────────────────────────────────────────")
        w(f"  Shallow CNN (2 layers, 16/32f):   acc=0.9886, eff=0.8935")
        w(f"  Deep CNN    (4 layers, 8/8/16/16): acc={test_acc:.4f}, eff={np.mean(effs):.4f}  <-- this model")
        w(f"  TCN         (4 layers + residual): acc=0.9899, eff=0.9137")
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


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description="Train deep dilated CNN (ablation: TCN without residuals)")
    parser.add_argument("--data-dir",   type=str, required=True)
    parser.add_argument("--epochs",     type=int,   default=500)
    parser.add_argument("--batch-size", type=int,   default=256)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--output-dir", type=str,   default="training_output_cnn_deep")
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_global_determinism(args.seed)
    print(f"  Seed {args.seed} — deterministic mode enabled")

    config = {
        "script":         "train_jump_cnn_deep.py",
        "purpose":        "Ablation: TCN conv layers without residual connections",
        "data_dir":       args.data_dir,
        "output_dir":     args.output_dir,
        "epochs":         args.epochs,
        "batch_size":     args.batch_size,
        "lr":             args.lr,
        "seed":           args.seed,
        "conv_configs":   CONV_CONFIGS,
        "kernel_size":    KERNEL_SIZE,
        "input_length":   INPUT_LENGTH,
        "input_channels": INPUT_CHANNELS,
    }
    with open(os.path.join(args.output_dir, "config.json"), "w") as cf:
        json.dump(config, cf, indent=2)

    # ── Load data ──
    print("Loading data...")
    X, y, templids = load_data(args.data_dir)
    print(f"  Total: {len(X)} ({int(y.sum())} jump, {int(len(y)-y.sum())} no-jump)")

    # ── Split (same seed as all other models → same test set) ──
    idx = np.arange(len(X))
    idx_trainval, idx_test = train_test_split(
        idx, test_size=0.15, random_state=args.seed, stratify=y)
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=0.176, random_state=args.seed, stratify=y[idx_trainval])

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
    np.save(os.path.join(args.output_dir, "norm_std.npy"), std)

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
            restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=7,
            min_lr=1e-6, verbose=1),
        callbacks.ModelCheckpoint(
            os.path.join(args.output_dir, "best_model.h5"),
            monitor="val_loss", save_best_only=True, verbose=1),
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
                                target_names=["No Jump", "Jump"], zero_division=0))

    qubit_results = compute_qubit_efficiency(y_test, y_pred, templids_test)
    print_qubit_efficiency_table(qubit_results)

    # ── Plots ──
    plot_training_history(history, args.output_dir)
    plot_roc_and_pr(y_test, y_prob, args.output_dir)
    plot_confusion_matrix(y_test, y_pred, args.output_dir)

    # ── Save ──
    save_results_log(args.output_dir, args, model, history,
                     test_loss, test_acc, y_test, y_pred, y_prob,
                     qubit_results,
                     y_train, y_val, templids_train, templids_val, templids_test)

    model.save(os.path.join(args.output_dir, "jump_cnn_deep_float32.h5"))
    print(f"\nModel saved to {args.output_dir}/jump_cnn_deep_float32.h5")

    # ── Summary ──
    trainable    = sum(tf.size(w).numpy() for w in model.trainable_weights)
    nontrainable = sum(tf.size(w).numpy() for w in model.non_trainable_weights)
    effs = [e for e, _, _ in qubit_results.values() if not np.isnan(e)]
    print("\n" + "=" * 65)
    print("ABLATION RESULT — Deep CNN (TCN without residuals)")
    print("=" * 65)
    print(f"  This model:   acc={test_acc:.4f}, eff={np.mean(effs):.4f}")
    print(f"  Shallow CNN:  acc=0.9886, eff=0.8935  (2 layers, 16/32 filters)")
    print(f"  TCN:          acc=0.9899, eff=0.9137  (4 layers + residual)")
    print(f"  Parameters:   {trainable+nontrainable:,}")
    print("=" * 65)


if __name__ == "__main__":
    main()
