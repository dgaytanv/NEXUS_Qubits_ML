#!/usr/bin/env python3
"""
generate_scans.py
-----------------
Generate synthetic qubit scans (with optional charge jumps) and save as PNG + NPY.
Designed to run as a SLURM job array — each task handles a subset of templates.

Usage:
    python generate_scans.py [options]

SLURM array example:
    sbatch --array=0-4 submit_generate_scans.sh \\
        --output-dir /home/export/jgaytanv/quantum/Data_Output_20k/ \\
        --n-plots 4000 \\
        --f 0.1 \\
        --noise-amplitude 0.02 \\
        --target-jump-size 0.1 \\
        --target-jump-size-max 0.5 \\
        --jump-step 0.01

Changes from original:
    - Added --target-jump-size-max: if set, jump size is sampled uniformly in
      [target-jump-size, target-jump-size-max] for each jump scan independently.
    - Each scan is now saved as both .png and .npy (dict with Q1_scan, vsweep,
      metadata) so the training scripts can load them directly.
"""

import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use('Agg')   # must be set before importing pyplot (no display on SLURM)
import matplotlib.pyplot as plt
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# Core functions
# ══════════════════════════════════════════════════════════════════════════════

def apply_jump(scan, i_jump, target_jump_size=None, jump_step=None):
    """
    Apply a charge jump to a scan at position i_jump.

    Parameters
    ----------
    scan : np.ndarray
        Input signal.
    i_jump : int
        Index at which the discontinuity is introduced.
    target_jump_size : float or None
        Desired jump magnitude in ADC units. If None, maximises discontinuity.
    jump_step : float or None
        Increment used to widen the search window if no exact match is found.

    Returns
    -------
    scan : np.ndarray
        Signal with jump applied.
    achieved_jump_size : float
        Actual jump magnitude of the chosen splice point.
    """
    Q1_longtempl = np.tile(scan, 2)
    value_before = Q1_longtempl[i_jump - 1] if i_jump > 0 else Q1_longtempl[0]

    local_slope = (Q1_longtempl[i_jump] - Q1_longtempl[i_jump - 1]
                   if 0 < i_jump < len(scan) - 1 else 0)

    max_rep       = len(Q1_longtempl) - (len(scan) - i_jump)
    valid_reps    = np.arange(max_rep)
    values_after  = Q1_longtempl[valid_reps]
    jump_sizes    = np.abs(values_after - value_before)
    signed_jumps  = values_after - value_before

    # Direction filter: jump should oppose local slope
    if local_slope > 0:
        direction_mask = signed_jumps < 0
    elif local_slope < 0:
        direction_mask = signed_jumps > 0
    else:
        direction_mask = np.ones(len(valid_reps), dtype=bool)

    filtered_reps       = valid_reps[direction_mask]
    filtered_jump_sizes = jump_sizes[direction_mask]

    # Fall back to full pool if filtered pool is empty or can't reach the target
    if (len(filtered_reps) == 0 or
            (target_jump_size is not None and
             filtered_jump_sizes.max() < target_jump_size)):
        filtered_reps       = valid_reps
        filtered_jump_sizes = jump_sizes

    if target_jump_size is not None:
        if jump_step is None:
            exact_matches = filtered_reps[np.isclose(filtered_jump_sizes, target_jump_size)]
            if len(exact_matches) == 0:
                raise ValueError(
                    f"No rep found for exact jump {target_jump_size:.4f}. "
                    f"Range: [{filtered_jump_sizes.min():.4f}, "
                    f"{filtered_jump_sizes.max():.4f}]. "
                    f"Consider setting jump_step."
                )
            best_rep           = exact_matches[0]
            achieved_jump_size = float(jump_sizes[best_rep])
        else:
            current_target = target_jump_size
            best_rep       = None

            while current_target <= filtered_jump_sizes.max():
                window_mask = (
                    (filtered_jump_sizes >= current_target) &
                    (filtered_jump_sizes <  current_target + jump_step)
                )
                window = filtered_reps[window_mask]
                if len(window) > 0:
                    best_rep = window[
                        np.argmin(np.abs(jump_sizes[window] - current_target))
                    ]
                    achieved_jump_size = float(jump_sizes[best_rep])
                    break
                current_target += jump_step

            if best_rep is None:
                raise ValueError(
                    f"No rep found in [{target_jump_size:.4f}, "
                    f"{filtered_jump_sizes.max():.4f}] "
                    f"with jump_step={jump_step:.4f}."
                )
    else:
        best_rep           = filtered_reps[np.argmax(filtered_jump_sizes)]
        achieved_jump_size = float(jump_sizes[best_rep])

    part1 = Q1_longtempl[:i_jump]
    part2 = Q1_longtempl[best_rep:best_rep + len(scan) - i_jump]
    scan  = np.concatenate([part1, part2])
    return scan, achieved_jump_size


def new_templ(templ, i_jump, rep, n_inputs):
    """Shift template so next scan begins where the last iteration ended."""
    out = np.empty_like(templ)
    for i in range(n_inputs):
        out[i] = templ[(i + i_jump - rep) % n_inputs]
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Argument parsing
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description='Generate synthetic qubit scans.')

    # Paths
    parser.add_argument('--basedir', type=str,
                        default='/home/export/jgaytanv/quantum/NEXUS_Qubits/',
                        help='Base directory for data and templates.')
    parser.add_argument('--output-dir', type=str,
                        default='/home/export/jgaytanv/quantum/Data_Output/',
                        help='Root output directory. A per-job subdir is created inside.')

    # Generation parameters
    parser.add_argument('--n-plots', type=int, default=100,
                        help='Number of scans to generate per template.')
    parser.add_argument('--f', type=float, default=0.5,
                        help='Fraction of scans that contain a jump (0.0–1.0).')
    parser.add_argument('--noise-amplitude', type=float, default=0.03,
                        help='Std dev of Gaussian noise added to each scan.')
    parser.add_argument('--target-jump-size', type=float, default=0.1,
                        help='Target jump magnitude in ADC units (minimum if '
                             '--target-jump-size-max is also set).')
    parser.add_argument('--target-jump-size-max', type=float, default=None,
                        help='If set, jump size is sampled uniformly in '
                             '[target-jump-size, target-jump-size-max] '
                             'independently for each jump scan. '
                             'If not set, all jumps use --target-jump-size exactly.')
    parser.add_argument('--jump-step', type=float, default=0.01,
                        help='Search step size when exact jump match not found.')

    # Plot labels
    parser.add_argument('--x-label', type=str, default='Charge Bias Voltage (V)')
    parser.add_argument('--y-label', type=str, default='Amplitude (ADC units)')

    # SLURM / reproducibility
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed. Defaults to SLURM_ARRAY_TASK_ID if set, else 0.')
    parser.add_argument('--templ-idx', type=int, default=None,
                        help='Process only this template index. '
                             'Defaults to SLURM_ARRAY_TASK_ID if set, else all templates.')

    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # ── SLURM array task ID ───────────────────────────────────────────────────
    slurm_task_id = os.environ.get('SLURM_ARRAY_TASK_ID', None)
    task_id       = int(slurm_task_id) if slurm_task_id is not None else 0

    # ── Seed ──────────────────────────────────────────────────────────────────
    seed = args.seed if args.seed is not None else task_id
    np.random.seed(seed)
    print(f"[Task {task_id}] Random seed: {seed}")

    # ── Paths ─────────────────────────────────────────────────────────────────
    template_datapath = os.path.join(args.basedir, 'SimulatedData_Gen', 'Templates', '')
    vsweep_path       = os.path.join(args.basedir, 'Data', 'vsweep.npy')
    output_dir        = os.path.join(args.output_dir, f'job_{task_id:04d}')
    os.makedirs(output_dir, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    all_files = sorted(glob.glob(os.path.join(template_datapath, '*templ_full*.npy')))
    if not all_files:
        print(f"[Task {task_id}] ERROR: No template files found in {template_datapath}")
        sys.exit(1)

    all_scans = [np.load(f, allow_pickle=True) for f in all_files]
    vsweep    = np.load(vsweep_path, allow_pickle=True)

    print(f"[Task {task_id}] Found {len(all_scans)} templates.")

    # ── Log jump size mode ────────────────────────────────────────────────────
    if args.target_jump_size_max is not None:
        print(f"[Task {task_id}] Jump size: uniform in "
              f"[{args.target_jump_size:.3f}, {args.target_jump_size_max:.3f}]")
    else:
        print(f"[Task {task_id}] Jump size: fixed at {args.target_jump_size:.3f}")

    # ── Determine which templates this task processes ─────────────────────────
    if args.templ_idx is not None:
        templ_indices = [args.templ_idx]
    elif slurm_task_id is not None:
        # Each SLURM task handles one template (array index = template index)
        templ_indices = [task_id]
    else:
        templ_indices = list(range(len(all_scans)))

    # ── Generate scans ────────────────────────────────────────────────────────
    total_saved = 0
    for templ_idx in templ_indices:
        if templ_idx >= len(all_scans):
            print(f"[Task {task_id}] WARNING: templ_idx={templ_idx} out of range, skipping.")
            continue

        Q1_templ_original = all_scans[templ_idx]
        n_inputs          = len(Q1_templ_original)
        vsweep_scan       = vsweep[:n_inputs]

        for plot_idx in range(args.n_plots):

            # ── Phase shift ──
            max_shift    = n_inputs // 10
            random_shift = np.random.randint(-max_shift, max_shift)
            Q1_shifted   = np.roll(Q1_templ_original, random_shift)

            # ── Gaussian noise ──
            noise   = np.random.normal(0, args.noise_amplitude, Q1_shifted.shape)
            Q1_scan = Q1_shifted + noise

            # ── Jump ──
            do_jump = int(np.random.choice([0, 1], p=[1 - args.f, args.f]))
            i_jump  = -1
            achieved_jump = 0.0

            if do_jump:
                i_jump = np.random.randint(0, n_inputs)

                # Sample jump size: uniform in [min, max] if range given, fixed otherwise
                if args.target_jump_size_max is not None:
                    sampled_jump_size = np.random.uniform(
                        args.target_jump_size, args.target_jump_size_max
                    )
                else:
                    sampled_jump_size = args.target_jump_size
                # was: Q1_scan, achieved_jump = apply_jump(
                #          Q1_scan, i_jump, args.target_jump_size, args.jump_step)

                Q1_scan, achieved_jump = apply_jump(
                    Q1_scan, i_jump, sampled_jump_size, args.jump_step
                )
                jump_str = (f'j{i_jump:03d}'
                            f'_v{vsweep_scan[i_jump]:.3f}'
                            f'_js{achieved_jump:.3f}')
            else:
                jump_str = 'jNone'

            # ── Base filename (shared by .png and .npy) ──
            fname_base = (f'templ{templ_idx:03d}'
                          f'_scan{plot_idx:04d}'
                          f'_jump{do_jump}'
                          f'_{jump_str}')

            # ── Save PNG ──
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(vsweep_scan, Q1_scan[:n_inputs], color='red', linewidth=1.5)
            ax.set_xlabel(args.x_label)
            ax.set_ylabel(args.y_label)
            ax.grid(True, alpha=0.3)
            fig.savefig(os.path.join(output_dir, fname_base + '.png'),
                        dpi=150, bbox_inches='tight')
            plt.close(fig)

            # ── Save NPY ──
            np.save(
                os.path.join(output_dir, fname_base + '.npy'),
                {
                    'Q1_scan':        Q1_scan[:n_inputs],
                    'vsweep':         vsweep_scan,
                    'templ_idx':      templ_idx,
                    'plot_idx':       plot_idx,
                    'random_shift':   random_shift,
                    'do_jump':        bool(do_jump),
                    'i_jump':         i_jump,
                    'achieved_jump':  achieved_jump,
                    'noise_amplitude': args.noise_amplitude,
                }
            )

            total_saved += 1
            print(f"[Task {task_id}] Saved {fname_base}  "
                  f"(shift={random_shift}, jump={do_jump}, i_jump={i_jump}, "
                  f"js={achieved_jump:.3f})")

    print(f"\n[Task {task_id}] Done. {total_saved} scans saved to {output_dir}")


if __name__ == '__main__':
    main()
