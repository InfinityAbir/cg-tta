"""Partition counts for the paper's partition table.

Reads the Camelyon17-WILDS metadata.csv (not redistributed; pass its path) and
writes tables/partition_counts.csv.  The CSV is shipped, so `make_supplement_tables.py`
does not need the dataset.

    python code/partition_counts.py /path/to/camelyon17_v1.0/metadata.csv
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REL = os.path.dirname(HERE)
OUT = os.environ.get("CG_TTA_OUT", os.path.join(REL, "tables"))

meta = pd.read_csv(sys.argv[1])
parts = [
    ("Source train", "0, 3, 4", meta[meta.center.isin([0, 3, 4]) & (meta.split == 0)]),
    ("Source ID val", "0, 3, 4", meta[meta.center.isin([0, 3, 4]) & (meta.split == 1)]),
    ("OOD validation", "1", meta[meta.center == 1]),
    ("Target (Center 2)", "2", meta[meta.center == 2]),
    ("Target (Center 0, 2nd rotation)", "0", meta[meta.center == 0]),
]
rows = []
for name, centers, df in parts:
    rows.append(dict(partition=name, centers=centers, patches=len(df),
                     slides=int(df.slide.nunique()), patients=int(df.patient.nunique()),
                     tumor=int(df.tumor.sum()), non_tumor=int((1 - df.tumor).sum())))
pd.DataFrame(rows).to_csv(os.path.join(OUT, "partition_counts.csv"), index=False)
print(pd.DataFrame(rows).to_string(index=False))
