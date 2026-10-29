"""
utils/data_loader.py
====================
PyTorch Dataset and DataLoader construction for FSDD.
Accepts explicit n_mfcc parameter so Task A (40) and Task B (13) both work
correctly without any config mutation.
"""

import os
import re

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import config
from utils.data_preprocessing import extract_features, pad_or_truncate, compute_stats


def load_file_paths(recordings_dir: str = None) -> tuple:
    if recordings_dir is None:
        recordings_dir = config.RECORDINGS_DIR
    paths, labels = [], []
    pattern = re.compile(r'^(\d)_')
    for fname in sorted(os.listdir(recordings_dir)):
        if not fname.endswith('.wav'):
            continue
        m = pattern.match(fname)
        if m:
            paths.append(os.path.join(recordings_dir, fname))
            labels.append(int(m.group(1)))
    return paths, labels


class SpokenDigitDataset(Dataset):
    """
    Each item: normalised MFCC feature matrix + digit label.
    Stores audio params explicitly so there is no dependency on global config.
    """

    def __init__(self, file_paths, labels, mean, std,
                 n_mfcc, sample_rate, n_fft, hop_length, max_len):
        self.file_paths  = file_paths
        self.labels      = labels
        self.mean        = mean
        self.std         = std
        self.n_mfcc      = n_mfcc
        self.sample_rate = sample_rate
        self.n_fft       = n_fft
        self.hop_length  = hop_length
        self.max_len     = max_len

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        feat = extract_features(self.file_paths[idx], self.n_mfcc,
                                self.sample_rate, self.n_fft, self.hop_length)
        feat = (feat - self.mean) / (self.std + 1e-8)   # normalise FIRST
        feat = pad_or_truncate(feat, self.max_len)       # pad AFTER (zeros stay zero)
        return (torch.tensor(feat, dtype=torch.float32),
                torch.tensor(self.labels[idx], dtype=torch.long))


def build_loaders(n_mfcc: int = None, recordings_dir: str = None,
                  batch_size: int = None, num_workers: int = 2,
                  random_state: int = 42):
    """
    Build train/val/test DataLoaders.

    Parameters
    ----------
    n_mfcc : int, optional
        Number of MFCC coefficients. Defaults to config.N_MFCC.
        Pass 13 for the constrained Task B model, 40 for Task A.
    """
    if n_mfcc       is None: n_mfcc       = config.N_MFCC
    if recordings_dir is None: recordings_dir = config.RECORDINGS_DIR
    if batch_size   is None: batch_size   = config.BATCH_SIZE

    # Audio params from config (these never change between tasks)
    sr  = config.SAMPLE_RATE
    nfft = config.N_FFT
    hop  = config.HOP_LENGTH
    mlen = config.MAX_LEN

    paths, labels = load_file_paths(recordings_dir)
    print(f"Found {len(paths)} recordings, {len(set(labels))} classes")
    print(f"Using n_mfcc={n_mfcc}  ->  feature_dim={3 * n_mfcc}")

    train_paths, tmp_paths, train_labels, tmp_labels = train_test_split(
        paths, labels, test_size=0.2, stratify=labels, random_state=random_state
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        tmp_paths, tmp_labels, test_size=0.5, stratify=tmp_labels,
        random_state=random_state
    )

    print("Computing feature statistics on training split...")
    mean, std = compute_stats(train_paths, n_mfcc, sr, nfft, hop)
    feature_dim = mean.shape[0]
    print(f"Stats: mean={mean.mean():.4f}, std={std.mean():.4f}")

    def make_ds(fp, lb):
        return SpokenDigitDataset(fp, lb, mean, std, n_mfcc, sr, nfft, hop, mlen)

    train_ds = make_ds(train_paths, train_labels)
    val_ds   = make_ds(val_paths,   val_labels)
    test_ds  = make_ds(test_paths,  test_labels)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)

    return train_loader, val_loader, test_loader, feature_dim
