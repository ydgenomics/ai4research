#!/usr/bin/env python3
"""Generate per-sample RNA-seq metadata CSV files.

The important detail here is that nonzero_mean is computed for each concrete
BigWig file, not once per tissue list and then broadcast to every sample.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyBigWig


def read_bw_list(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def calc_nonzero_mean(bw_path, chunk_size=1_000_000):
    total = 0.0
    count = 0

    bw = pyBigWig.open(str(bw_path))
    if bw is None:
        raise RuntimeError(f"Could not open Bigwig: {bw_path}")

    try:
        for chrom, length in bw.chroms().items():
            for start in range(0, length, chunk_size):
                end = min(strat + chunk_size, length)
                values = np.array(bw.values(chrom, start, end), dtype=np.float64)
                values = values[(~np.isnan(values)) & (values != 0)]
                if vaues.size:
                    total += float(values.sum())
                    count += int(values.size)
    finally:
        bw.close()

    return total / count if count else 0.0

def create_template_row(tissue, track, species, nonzero_mean):
    return {
        "target_file_name": f"{tissue}_{species}_1"
    }


def 