#!/usr/bin/env python3
"""
QAT for the deep dilated CNN (4 layers, no residual connections).

Ablation: does the deep CNN without residuals survive quantization
as well as the TCN with residuals?

Architecture (matches train_jump_cnn_deep.py):
    Input(128, 2)
    → QConv1D(8,  k=3, dil=1) → ReLU → BN
    → QConv1D(8,  k=3, dil=1) → ReLU → BN
    → QConv1D(16, k=3, dil=2) → ReLU → BN
    → QConv1D(16, k=3, dil=2) → ReLU → BN
    → AvgPool(128) → Flatten → QDense(1) → Sigmoid

Usage:
    python train_jump_cnn_deep_qat.py \
        --data-dir /home/export/jgaytanv/quantum/Data_Output_100k/flat \
        --pretrained training_output_cnn_deep_100k/best_model.h5 \
        --output-dir training_output_cnn_deep_qat
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
from qkeras import QConv1D, QDense, QActivation, quantized_bits
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


# ── Architecture ──
INPUT_LENGTH   = 128
INPUT_CHANNELS = 2
CONV_CONFIGS = [
    {"filters": 8,  "dilation": 1},
    {"filters": 8,  "dilation": 1},
    {"filters": 16, "dilation": 2},
    {"filters": 16, "dilation": 2},
]
KERNEL_SIZE = 3

# ── Quantization ──
TOTAL_BITS = 16
INT_BITS   = 6

QUBIT_MAP = {0: "Q1", 1: "Q2", 2: "Q3", 3: "Q4 (SC)", 4: "Q4 (SO)"}

# ── Wilen-style magnitude binning ──
# Bin edges in units of e for |Δq| ∈ [0.1, 0.5].
#JUMP_MAG_BIN_EDGES = (0.1, 0.2, 0.3, 0.4, 0.5)
# 15 bins of equal width across [0.1, 0.5] e, matching Wilen et al. Table S2.
JUMP_MAG_BIN_EDGES = tuple(np.linspace(0.1, 0.5, 16))

def build_qat_model():
    """Build the QKeras quantized deep CNN."""
    qb = quantized_bits(bits=TOTAL_BITS, integer=INT_BITS, alpha=1)

    x = inp = layers.Input(shape=(INPUT_LENGTH, INPUT_CHANNELS), name="inp_signal")

    for i, cfg in enumerate(CONV_CONFIGS):
        x = QConv1D(
            cfg["filters"], KERNEL_SIZE,
            dilation_rate=cfg["dilation"],
            padding="causal",
            use_bias=True,
            kernel_initializer="he_normal",
            kernel_quantizer=qb,
            bias_quantizer=qb,
            name=f"conv1d_{i}",
        )(x)
        x = layers.ReLU(name=f"relu_{i}")(x)
        x = layers.BatchNormalization(name=f"bn_{i}")(x)
        # Clip activations to ap_fixed<16,6> range (same as TCN QAT)
        x = QActivation(quantized_bits(bits=TOTAL_BITS, integer=INT_BITS, alpha=1),
                         name=f"qclip_{i}")(x)

    x = layers.AveragePooling1D(pool_size=INPUT_LENGTH, name="avgpool")(x)
    x = layers.Flatten(name="flatten")(x)
    x = QDense(1, kernel_quantizer=qb, bias_quantizer=qb, name="dense_out")(x)
    x = layers.Activation("sigmoid", name="sigmoid")(x)

    return keras.Model(inputs=inp, outputs=x, name="jump_cnn_deep_qat")


def build_float_model():
    """Rebuild the float32 model for loading pretrained weights."""
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

    return keras.Model(inputs=inp, outputs=x, name="jump_cnn_deep")


def transfer_weights(float_model, qat_model):
    transferred, skipped = 0, 0
    float_weights = {w.name: w for w in float_model.weights}
    for qw in qat_model.weights:
        if qw.name in float_weights:
            fw = float_weights[qw.name]
            if qw.shape == fw.shape:
                qw.assign(fw)
                transferred += 1
            else:
                print(f"  Shape mismatch: {qw.name} QAT={qw.shape} float={fw.shape}")
                skipped += 1
        else:
            skipped += 1
    print(f"  Transferred {transferred} weight tensors, skipped {skipped}")
    return transferred


# ── Data loading ──

def load_data(data_dir):
    X, y, templids, jump_mags = [], [], [], []
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
        # Achieved jump magnitude (0.0 for no-jump signals or older files
        # without this field)
        jump_mags.append(float(data.get("achieved_jump", 0.0)))
    print(f"  Found {n_jump} jump files and {n_nojump} no-jump files")
    return (np.array(X,         dtype=np.float32),
            np.array(y,         dtype=np.float32),
            np.array(templids,  dtype=np.int32),
            np.array(jump_mags, dtype=np.float32))


def normalize(X_train, X_val, X_test):
    mean = X_train.mean(axis=0, keepdims=True)
    std  = X_train.std(axis=0, keepdims=True) + 1e-8
    return (X_train-mean)/std, (X_val-mean)/std, (X_test-mean)/std, mean, std


def compute_qubit_efficiency(y_test, y_pred, templids_test):
    """Existing metric: per-qubit efficiency over all jump magnitudes."""
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


def compute_qubit_efficiency_by_magnitude(y_test, y_pred, templids_test,
                                           jump_mags_test,
                                           bin_edges=JUMP_MAG_BIN_EDGES,
                                           min_samples_per_bin=1):
    """
    Additional metric: per-qubit efficiency binned by jump magnitude.

    Reports mean and std across magnitude bins, matching the error definition
    used in Wilen et al. (Table S2): the standard deviation captures
    systematic variation in detection efficiency across the jump-magnitude
    range, rather than qubit-to-qubit variation.
    """
    bin_edges = np.asarray(bin_edges)
    n_bins    = len(bin_edges) - 1
    results   = {}
    for tidx, qname in QUBIT_MAP.items():
        eff_per_bin = []
        bin_counts  = []
        for b in range(n_bins):
            mask = ((templids_test == tidx) & (y_test == 1) &
                    (jump_mags_test >= bin_edges[b]) &
                    (jump_mags_test <  bin_edges[b + 1]))
            n_total = int(mask.sum())
            if n_total < min_samples_per_bin:
                continue
            eff_per_bin.append(float((y_pred[mask] == 1).mean()))
            bin_counts.append(n_total)
        if len(eff_per_bin) >= 2:
            results[qname] = (float(np.mean(eff_per_bin)),
                              float(np.std(eff_per_bin)),
                              bin_counts)
        else:
            results[qname] = (float('nan'), float('nan'), bin_counts)
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


def print_qubit_efficiency_by_magnitude(results, bin_edges=JUMP_MAG_BIN_EDGES):
    print("\n" + "=" * 70)
    print("PER-QUBIT EFFICIENCY — binned by |Δq|  (Wilen et al. Table S2 style)")
    print("=" * 70)
    print(f"  Bin edges (e): {list(map(float, bin_edges))}")
    print(f"  {'Qubit':<12} {'Mean ± syst':>18}  {'bin counts':>26}")
    print("  " + "-" * 60)
    means = []
    for qname, (mean, syst, counts) in results.items():
        if np.isnan(mean):
            print(f"  {qname:<12} {'N/A':>18}  {str(counts):>26}")
        else:
            means.append(mean)
            print(f"  {qname:<12} {mean:>10.4f} ± {syst:.4f}  {str(counts):>26}")
    if means:
        print("  " + "-" * 60)
        print(f"  {'Overall':<12} {np.mean(means):>10.4f} ± {np.std(means):.4f}  (mean ± std across qubits)")
    print("=" * 70)


# ── Plots ──

def plot_training_history(history, output_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history["loss"],     label="Train")
    ax1.plot(history.history["val_loss"], label="Val")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("QAT — Binary Cross-Entropy Loss"); ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.plot(history.history["accuracy"],     label="Train")
    ax2.plot(history.history["val_accuracy"], label="Val")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.set_title("QAT — Classification Accuracy"); ax2.legend(); ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "qat_training_curves.png"), dpi=150)
    plt.close(fig)

def plot_roc_and_pr(y_true, y_prob, output_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    ax1.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    ax1.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax1.set_xlabel("FPR"); ax1.set_ylabel("TPR")
    ax1.set_title("ROC Curve (QAT)"); ax1.legend(); ax1.grid(True, alpha=0.3)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    ax2.plot(recall, precision, color="#ff7f0e", lw=2, label=f"PR (AUC = {pr_auc:.4f})")
    ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
    ax2.set_title("PR Curve (QAT)"); ax2.legend(); ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "qat_roc_pr_curves.png"), dpi=150)
    plt.close(fig)

def plot_confusion_matrix(y_true, y_pred, output_dir):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Jump", "Jump"]); ax.set_yticklabels(["No Jump", "Jump"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion Matrix (QAT)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=16, fontweight="bold")
    fig.colorbar(im); fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "qat_confusion_matrix.png"), dpi=150)
    plt.close(fig)


# ── Results logger ──

def save_results_log(output_dir, args, model, history,
                     test_loss, test_acc, y_test, y_pred, y_prob,
                     qubit_results, qubit_results_mag,
                     float_test_acc, float_efficiency):
    import datetime
    trainable    = sum(tf.size(w).numpy() for w in model.trainable_weights)
    nontrainable = sum(tf.size(w).numpy() for w in model.non_trainable_weights)
    fpr, tpr, _  = roc_curve(y_test, y_prob)
    roc_auc      = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    pr_auc       = auc(rec, prec)
    best_val_loss = min(history.history["val_loss"])
    best_val_acc  = max(history.history["val_accuracy"])
    n_epochs_run  = len(history.history["loss"])
    effs = [e for e, _, _ in qubit_results.values() if not np.isnan(e)]
    mean_eff = np.mean(effs) if effs else 0.0

    with open(os.path.join(output_dir, "results.txt"), "w") as f:
        def w(line=""):
            f.write(line + "\n")
        w("=" * 65)
        w("QAT RESULTS — Deep Dilated CNN (train_jump_cnn_deep_qat.py)")
        w("=" * 65)
        w(f"  Timestamp:      {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        w(f"  Output dir:     {output_dir}")
        w(f"  Pretrained:     {args.pretrained}")
        w(f"  Quantization:   ap_fixed<{TOTAL_BITS},{INT_BITS}>")
        w()
        w("── Purpose ───────────────────────────────────────────────────")
        w(f"  Ablation: does deep CNN survive quantization without residuals?")
        w()
        w("── Float32 baseline ──────────────────────────────────────────")
        w(f"  Float32 test acc:       {float_test_acc:.4f}")
        w(f"  Float32 efficiency:     {float_efficiency:.4f}")
        w()
        w("── QAT config ────────────────────────────────────────────────")
        w(f"  epochs:         {args.epochs}  (ran {n_epochs_run})")
        w(f"  batch-size:     {args.batch_size}")
        w(f"  lr:             {args.lr}")
        w(f"  seed:           {args.seed}")
        w()
        w("── Architecture ──────────────────────────────────────────────")
        w(f"  Input:          ({INPUT_LENGTH}, {INPUT_CHANNELS})")
        for i, cfg in enumerate(CONV_CONFIGS):
            w(f"  QConv1D_{i}:    f={cfg['filters']}, k={KERNEL_SIZE}, dil={cfg['dilation']}")
            w(f"  ReLU_{i} -> BN_{i} -> clip ±32")
        w(f"  AvgPool(128) -> Flatten -> QDense(1) -> Sigmoid")
        w(f"  Parameters (trainable):     {trainable:,}")
        w(f"  Parameters (non-trainable): {nontrainable:,}")
        w()
        w("── QAT results ───────────────────────────────────────────────")
        w(f"  Best val loss:     {best_val_loss:.4f}")
        w(f"  Best val accuracy: {best_val_acc:.4f}")
        w(f"  Test loss:         {test_loss:.4f}")
        w(f"  Test accuracy:     {test_acc:.4f}")
        w(f"  ROC AUC:           {roc_auc:.4f}")
        w(f"  PR AUC:            {pr_auc:.4f}")
        w()
        w("── Accuracy degradation ──────────────────────────────────────")
        w(f"  Float32 acc:  {float_test_acc:.4f}")
        w(f"  QAT acc:      {test_acc:.4f}")
        w(f"  Delta:        {test_acc - float_test_acc:+.4f}")
        w()
        w("── Classification Report ─────────────────────────────────────")
        w(classification_report(y_test, y_pred,
                                target_names=["No Jump", "Jump"], zero_division=0))
        w("── Per-Qubit Efficiency ──────────────────────────────────────")
        w(f"  {'Qubit':<12} {'Efficiency':>12}  {'Correct / Total':>18}")
        w("  " + "-" * 46)
        for qname, (eff, nc, nt) in qubit_results.items():
            if np.isnan(eff):
                w(f"  {qname:<12} {'N/A':>12}  {'no samples':>18}")
            else:
                w(f"  {qname:<12} {eff:>11.4f}   {nc:>6d} / {nt:<6d}")
        if effs:
            w("  " + "-" * 46)
            w(f"  {'Overall':<12} {np.mean(effs):>11.4f} +/- {np.std(effs):.4f}")
        w()
        w("── Per-Qubit Efficiency by |Δq| (Wilen-style errors) ─────────")
        w(f"  Bin edges (e): {list(map(float, JUMP_MAG_BIN_EDGES))}")
        w(f"  {'Qubit':<12} {'Mean ± syst':>18}  {'bin counts':>26}")
        w("  " + "-" * 60)
        means_mag = []
        for qname, (mean, syst, counts) in qubit_results_mag.items():
            if np.isnan(mean):
                w(f"  {qname:<12} {'N/A':>18}  {str(counts):>26}")
            else:
                means_mag.append(mean)
                w(f"  {qname:<12} {mean:>10.4f} ± {syst:.4f}  {str(counts):>26}")
        if means_mag:
            w("  " + "-" * 60)
            w(f"  {'Overall':<12} {np.mean(means_mag):>10.4f} ± {np.std(means_mag):.4f}  (across qubits)")
        w()
        w("── Full ablation comparison (QAT ap_fixed<16,6>) ─────────────")
        w(f"  Shallow CNN QAT (2 layers):      acc=0.9600, eff=0.7393")
        w(f"  Deep CNN QAT    (4 layers):       acc={test_acc:.4f}, eff={mean_eff:.4f}  <-- this")
        w(f"  TCN QAT         (4 layers+resid): acc=0.9874, eff=0.8798")
        w()
        w("=" * 65)

    print(f"  Saved results.txt")


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description="QAT for deep CNN (ablation)")
    parser.add_argument("--data-dir",    type=str, required=True)
    parser.add_argument("--pretrained",  type=str, required=True)
    parser.add_argument("--epochs",      type=int,   default=100)
    parser.add_argument("--batch-size",  type=int,   default=256)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--output-dir",  type=str,   default="training_output_cnn_deep_qat")
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_global_determinism(args.seed)
    print(f"  Seed {args.seed} — deterministic mode enabled")

    config = {
        "script":     "train_jump_cnn_deep_qat.py",
        "purpose":    "Ablation: deep CNN QAT without residuals",
        "pretrained": args.pretrained,
        "data_dir":   args.data_dir,
        "total_bits": TOTAL_BITS,
        "int_bits":   INT_BITS,
        "epochs":     args.epochs,
        "batch_size": args.batch_size,
        "lr":         args.lr,
        "seed":       args.seed,
        "jump_mag_bin_edges": list(JUMP_MAG_BIN_EDGES),
    }
    with open(os.path.join(args.output_dir, "config.json"), "w") as cf:
        json.dump(config, cf, indent=2)

    # ── Load data ──
    print("Loading data...")
    X, y, templids, jump_mags = load_data(args.data_dir)
    print(f"  Total: {len(X)} ({int(y.sum())} jump, {int(len(y)-y.sum())} no-jump)")

    idx = np.arange(len(X))
    idx_trainval, idx_test = train_test_split(
        idx, test_size=0.15, random_state=args.seed, stratify=y)
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=0.176, random_state=args.seed, stratify=y[idx_trainval])

    X_train, y_train = X[idx_train], y[idx_train]
    X_val,   y_val   = X[idx_val],   y[idx_val]
    X_test,  y_test  = X[idx_test],  y[idx_test]
    templids_test    = templids[idx_test]
    jump_mags_test   = jump_mags[idx_test]
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    total = n_neg + n_pos
    class_weight = {0: total / (2 * n_neg), 1: total / (2 * n_pos)}

    X_train, X_val, X_test, mean, std = normalize(X_train, X_val, X_test)
    np.save(os.path.join(args.output_dir, "norm_mean.npy"), mean)
    np.save(os.path.join(args.output_dir, "norm_std.npy"), std)

    # ── Float32 baseline ──
    print("\nLoading pretrained float32 model...")
    float_model = build_float_model()
    float_model.load_weights(args.pretrained)
    float_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    _, float_acc = float_model.evaluate(X_test, y_test, verbose=0)
    float_pred = (float_model.predict(X_test, verbose=0).flatten() > 0.5).astype(int)
    float_qr   = compute_qubit_efficiency(y_test, float_pred, templids_test)
    float_effs = [e for e, _, _ in float_qr.values() if not np.isnan(e)]
    float_eff  = np.mean(float_effs) if float_effs else 0.0
    print(f"  Float32 baseline — acc: {float_acc:.4f}, efficiency: {float_eff:.4f}")

    # ── Build QAT model ──
    print("\nBuilding QAT model...")
    qat_model = build_qat_model()
    qat_model.summary()

    print("\nTransferring pretrained weights...")
    transfer_weights(float_model, qat_model)

    qat_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    _, pre_acc = qat_model.evaluate(X_test, y_test, verbose=0)
    print(f"  QAT before fine-tuning — acc: {pre_acc:.4f} (float32: {float_acc:.4f})")

    # ── Fine-tune ──
    print(f"\nFine-tuning with QAT (lr={args.lr}, {args.epochs} epochs max)...")
    cb = [
        callbacks.EarlyStopping(
            monitor="val_loss", patience=30,
            restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=7,
            min_lr=1e-7, verbose=1),
        callbacks.ModelCheckpoint(
            os.path.join(args.output_dir, "best_model_qat.h5"),
            monitor="val_loss", save_best_only=True, verbose=1),
    ]

    history = qat_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=cb,
        class_weight=class_weight,
        verbose=1,
    )

    # ── Evaluate ──
    print("\nEvaluating QAT model...")
    test_loss, test_acc = qat_model.evaluate(X_test, y_test, verbose=0)
    print(f"  QAT test acc: {test_acc:.4f}  (float32: {float_acc:.4f}, delta: {test_acc-float_acc:+.4f})")

    y_prob = qat_model.predict(X_test, verbose=0).flatten()
    y_pred = (y_prob > 0.5).astype(int)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=["No Jump", "Jump"], zero_division=0))

    # Existing metric: per-qubit efficiency over all jump magnitudes
    qubit_results = compute_qubit_efficiency(y_test, y_pred, templids_test)
    print_qubit_efficiency_table(qubit_results)

    # New metric: per-qubit efficiency binned by jump magnitude (Wilen-style)
    qubit_results_mag = compute_qubit_efficiency_by_magnitude(
        y_test, y_pred, templids_test, jump_mags_test)
    print_qubit_efficiency_by_magnitude(qubit_results_mag)

    # ── Plots ──
    plot_training_history(history, args.output_dir)
    plot_roc_and_pr(y_test, y_prob, args.output_dir)
    plot_confusion_matrix(y_test, y_pred, args.output_dir)

    # ── Save ──
    save_results_log(args.output_dir, args, qat_model, history,
                     test_loss, test_acc, y_test, y_pred, y_prob,
                     qubit_results, qubit_results_mag,
                     float_acc, float_eff)

    qat_model.save(os.path.join(args.output_dir, "jump_cnn_deep_qat.h5"))
    print(f"\nQAT model saved to {args.output_dir}/jump_cnn_deep_qat.h5")

    effs = [e for e, _, _ in qubit_results.values() if not np.isnan(e)]
    print("\n" + "=" * 65)
    print("ABLATION — QAT COMPARISON")
    print("=" * 65)
    print(f"  Shallow CNN QAT:  acc=0.9600, eff=0.7393")
    print(f"  Deep CNN QAT:     acc={test_acc:.4f}, eff={np.mean(effs):.4f}  <-- this")
    print(f"  TCN QAT:          acc=0.9874, eff=0.8798")
    print("=" * 65)


if __name__ == "__main__":
    main()
