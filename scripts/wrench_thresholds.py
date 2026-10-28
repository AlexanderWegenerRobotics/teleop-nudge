"""Derives contact and excessive-wrench thresholds from logged FMB episodes.

    python scripts/wrench_thresholds.py --store C:/Users/ceti/Documents/05_data/fmb-1-desk

Reads observations/arm_*/F_ext (the avatar's estimate, what the engine sees live) and the MuJoCo
ground-truth contact force from arm_*_wrench_truth.csv, and prints percentiles of the estimated
force norm in free space, in contact, and at true contact onsets.
"""

from __future__ import annotations

import argparse
import glob
import os

import h5py
import numpy as np

PHASE = {0: "idle", 1: "approach", 2: "grasp", 3: "transport", 4: "place", 5: "insert", 6: "correct"}


def truth_contact(ep_dir: str, arm: str):
    """(sim_time, |F_contact|) from the ground-truth wrench csv, or None if the file is absent."""
    path = os.path.join(ep_dir, f"{arm}_wrench_truth.csv")
    if not os.path.exists(path):
        return None
    d = np.genfromtxt(path, delimiter=";", names=True, invalid_raise=False)
    if d.size == 0:
        return None
    f = np.sqrt(d["F_world_fx"] ** 2 + d["F_world_fy"] ** 2 + d["F_world_fz"] ** 2)
    return d["sim_time"], f


def arm_sim_time(ep_dir: str, arm: str, n: int):
    """sim_time per hdf5 row, from the 500 Hz arm csv resampled to the episode grid."""
    path = os.path.join(ep_dir, f"{arm}.csv")
    d = np.genfromtxt(path, delimiter=";", names=True, usecols=("time", "sim_time"), invalid_raise=False)
    return np.interp(np.linspace(d["time"][0], d["time"][-1], n), d["time"], d["sim_time"])


def pct(x, q=(50, 90, 95, 99)):
    x = np.asarray(x)
    return "n/a" if x.size == 0 else "  ".join(f"p{p}={np.percentile(x, p):5.1f}" for p in q) + f"  max={x.max():5.1f}  n={x.size}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--arm", default="arm_left")
    ap.add_argument("--truth-on-n", type=float, default=0.5, help="ground-truth force that counts as contact")
    args = ap.parse_args()

    free, contact, onset_est, onset_lag, per_phase = [], [], [], [], {k: [] for k in PHASE}
    truth_free = []
    for f in sorted(glob.glob(os.path.join(args.store, "*", "episode.hdf5"))):
        ep = os.path.dirname(f)
        with h5py.File(f) as h:
            key = f"observations/{args.arm}/F_ext"
            if key not in h:
                continue
            fe = h[key][:]
            pk = f"labels/{args.arm}_phase"
            phase = h[pk][:] if pk in h else np.full(len(fe), -1)
            state = h[f"observations/{args.arm}/state"][:]
            rate = float(h.attrs.get("rate_hz", 30.0))
        norm = np.linalg.norm(fe[:, :3], axis=1)
        engaged = state == 4
        for k in PHASE:
            per_phase[k].extend(norm[engaged & (phase == k)])
        tc = truth_contact(ep, args.arm)
        if tc is None:
            continue
        t_sim = arm_sim_time(ep, args.arm, len(norm))
        truth = np.interp(t_sim, tc[0], tc[1])
        in_contact = truth > args.truth_on_n
        free.extend(norm[engaged & ~in_contact])
        contact.extend(norm[engaged & in_contact])
        truth_free.extend(truth[engaged & ~in_contact])
        onsets = np.flatnonzero(np.diff(in_contact.astype(int)) == 1) + 1
        for i in onsets:
            if not engaged[i]:
                continue
            win = norm[i:i + int(0.3 * rate)]
            onset_est.append(win.max() if win.size else norm[i])
            above = np.flatnonzero(win > 3.0)
            onset_lag.append(above[0] / rate if above.size else np.nan)

    print(f"estimated |F_ext| [N], {args.arm}")
    print(f"  free space (truth < {args.truth_on_n} N):   {pct(free)}")
    print(f"  in contact:                 {pct(contact)}")
    print(f"  peak within 0.3 s of onset: {pct(onset_est)}")
    lag = np.asarray(onset_lag)
    print(f"  onsets: {lag.size}, estimate crosses 3 N within 0.3 s in {np.isfinite(lag).mean() * 100:.0f}%,"
          f" median lag {np.nanmedian(lag) * 1000:.0f} ms")
    print("  per phase:")
    for k, name in PHASE.items():
        print(f"    {name:10s} {pct(per_phase[k])}")
    print("\nsuggestion: on_n ≈ free-space p99 + margin, off_n ≈ free-space p90, warn_n ≈ contact p90, limit_n = reflex threshold")


if __name__ == "__main__":
    main()
