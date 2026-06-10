"""
dataset.py — RadioML 2018.01A data pipeline
Updated: RAM caching to fix slow epoch time
"""

import os
import time
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ── constants ─────────────────────────────────────────────────────────────────
MODULATIONS = [
    "OOK",       "4ASK",      "8ASK",      "BPSK",      "QPSK",
    "8PSK",      "16PSK",     "32PSK",     "16APSK",    "32APSK",
    "64APSK",    "128APSK",   "16QAM",     "32QAM",     "64QAM",
    "128QAM",    "256QAM",    "AM-SSB-WC", "AM-SSB-SC", "AM-DSB-WC",
    "AM-DSB-SC", "FM",        "GMSK",      "OQPSK",
]
SNR_VALUES = list(range(-20, 32, 2))
N_CLASSES  = len(MODULATIONS)
N_SNR      = len(SNR_VALUES)
N_PER_CELL = 4096


# ── dataset ───────────────────────────────────────────────────────────────────
class RadioMLDataset(Dataset):
    """
    PyTorch Dataset for RadioML 2018.01A with optional RAM caching.

    Parameters
    ----------
    hdf5_path : str
    indices   : np.ndarray   row indices for this split
    Y         : np.ndarray   class labels (all samples)
    Z         : np.ndarray   SNR labels (all samples)
    cache     : bool         if True, load all X into RAM at init.
                             Fast training but uses ~5 GB RAM for
                             a 638K-sample subset.
    """

    def __init__(self, hdf5_path, indices, Y, Z, cache=True):
        self.hdf5_path = hdf5_path
        self.indices   = np.sort(indices)
        self.labels    = Y[self.indices]
        self.snrs      = Z[self.indices]
        self.cache     = cache
        self.X_cache   = None

        if cache:
            print(f"Caching {len(self.indices):,} samples into RAM...")
            t0      = time.time()
            chunk   = 50_000
            n       = len(self.indices)
            x_parts = []

            with h5py.File(hdf5_path, "r") as f:
                for start in range(0, n, chunk):
                    end  = min(start + chunk, n)
                    rows = self.indices[start:end]
                    x_parts.append(f["X"][rows])
                    if start % 200_000 == 0:
                        print(f"  {end:,} / {n:,}")

            X_raw = np.concatenate(x_parts, axis=0)        # (N, 1024, 2)
            X_raw = X_raw.transpose(0, 2, 1).astype(np.float32)  # (N, 2, 1024)
            mean  = X_raw.mean(axis=2, keepdims=True)
            std   = X_raw.std(axis=2,  keepdims=True) + 1e-8
            self.X_cache = (X_raw - mean) / std

            elapsed = time.time() - t0
            ram_gb  = self.X_cache.nbytes / 1e9
            print(f"Cached {n:,} samples in {elapsed:.0f}s  ({ram_gb:.1f} GB RAM)")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        if self.X_cache is not None:
            x = torch.from_numpy(self.X_cache[i])
        else:
            with h5py.File(self.hdf5_path, "r") as f:
                x = f["X"][int(self.indices[i])]
            x    = x.T.astype(np.float32)
            mean = x.mean(axis=1, keepdims=True)
            std  = x.std(axis=1,  keepdims=True) + 1e-8
            x    = torch.from_numpy((x - mean) / std)

        return x, int(self.labels[i]), float(self.snrs[i])


# ── split builder ─────────────────────────────────────────────────────────────
def build_splits(Y, Z, samples_per_cell=1024):
    """
    Stratified split. samples_per_cell controls subset size.
    Default 1024 gives ~5.2 GB RAM — safe for Kaggle.
    Full 2048 gives ~10.4 GB — use only if RAM allows.
    """
    half = samples_per_cell // 2
    gen_idx, det_idx = [], []
    for cls in range(N_CLASSES):
        for snr in SNR_VALUES:
            rows = np.where((Y == cls) & (Z == snr))[0]
            gen_idx.append(rows[:half])
            det_idx.append(rows[half : half * 2])
    return (np.concatenate(gen_idx).astype(np.int32),
            np.concatenate(det_idx).astype(np.int32))


# ── convenience factory ───────────────────────────────────────────────────────
def get_loaders(hdf5_path, batch_size=512, num_workers=0,
                samples_per_cell=1024, cache=True):
    """
    Load Y and Z, build splits, cache X, return loaders.

    num_workers=0 is intentional — with cached RAM data,
    multiprocessing adds overhead rather than helping.
    """
    with h5py.File(hdf5_path, "r") as f:
        Y = np.argmax(f["Y"][:], axis=1).astype(np.int8)
        Z = f["Z"][:, 0].astype(np.int16)

    gen_idx, det_idx = build_splits(Y, Z, samples_per_cell)

    ds_gen = RadioMLDataset(hdf5_path, gen_idx, Y, Z, cache=cache)
    ds_det = RadioMLDataset(hdf5_path, det_idx, Y, Z, cache=cache)

    kw = dict(batch_size=batch_size, num_workers=num_workers,
              pin_memory=True, drop_last=True, shuffle=True)

    return (
        DataLoader(ds_gen, **kw),
        DataLoader(ds_det, **kw),
        dict(Y=Y, Z=Z, gen_idx=gen_idx, det_idx=det_idx,
             MODULATIONS=MODULATIONS, SNR_VALUES=SNR_VALUES),
    )
