import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

_CACHE = {}


def map_nsl_attack_id(aid: int) -> int:
    aid = int(aid)
    if aid == 0:
        return 0
    if aid in {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}:
        return 1
    if aid in {11, 12, 13, 14, 15}:
        return 2
    if aid in {16, 17, 18, 19, 20, 21}:
        return 3
    return 4


def _clean_nsl_line(line):
    parts = [p.strip() for p in line.rstrip("\n").split(",")]
    if len(parts) < 41:
        return None
    if len(parts) > 42:
        parts = parts[:41] + [",".join(parts[41:])]
    return parts


def _load_nsl_raw(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
    if not files:
        raise FileNotFoundError(f"No NSL-KDD files in {data_dir}")

    rows = []
    for f in files:
        with open(f, "r", encoding="utf-8", errors="ignore") as fh:
            for ln in fh:
                r = _clean_nsl_line(ln)
                if r is not None:
                    rows.append(r)

    if not rows:
        raise RuntimeError("NSL-KDD parsing produced no rows")

    return pd.DataFrame(rows)


def load_nsl_kdd_data(
    data_dir="data/nsl_kdd",
    target_dim=79,
    test_size=0.2,
    random_state=42,
):
    key = ("nsl", data_dir, target_dim)
    if key in _CACHE:
        return _CACHE[key]

    df = _load_nsl_raw(data_dir)

    if df.shape[1] >= 42:
        df = df.drop(columns=[df.columns[41]])

    label_col = df.columns[-2]
    feature_cols = df.columns[:-2]

    for idx in [1, 2, 3]:
        if idx < len(feature_cols):
            col = feature_cols[idx]
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    attack_ids = (
        pd.to_numeric(df[label_col], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    y = attack_ids.apply(map_nsl_attack_id).astype(np.int32).values
    if y.size == 0:
        raise RuntimeError("NSL-KDD produced empty labels")

    X = (
        df[feature_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .astype(np.float32)
        .values
    )

    if X.shape[1] < target_dim:
        X = np.hstack(
            [X, np.zeros((X.shape[0], target_dim - X.shape[1]), dtype=np.float32)]
        )
    else:
        X = X[:, :target_dim]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr).astype(np.float32)
    X_te = scaler.transform(X_te).astype(np.float32)

    _CACHE[key] = ((X_tr, y_tr), (X_te, y_te))
    print(f"✓ Loaded NSL-KDD: Train={X_tr.shape}, Test={X_te.shape}")

    return _CACHE[key]


def load_cicids2017_data(
    data_dir="data/cicids2017",
    target_dim=79,
    test_size=0.2,
    random_state=42,
    nrows=None,
):
    key = ("cic", data_dir, target_dim, nrows)
    if key in _CACHE:
        return _CACHE[key]

    csvs = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not csvs:
        raise FileNotFoundError(f"No CICIDS CSV files in {data_dir}")

    dfs = []
    for f in csvs:
        df_part = pd.read_csv(
            f, nrows=nrows, encoding="utf-8", encoding_errors="ignore"
        )
        df_part = df_part.replace([np.inf, -np.inf], np.nan)
        dfs.append(df_part)

    df = pd.concat(dfs, ignore_index=True)

    label_col = next((c for c in df.columns if "label" in c.lower()), None)
    if label_col is None:
        raise RuntimeError("CICIDS label column not found")

    def map_label(s):
        s = str(s).lower()
        if "benign" in s:
            return 0
        if "ddos" in s or "dos" in s:
            return 1
        if "port" in s:
            return 2
        if "brute" in s or "ftp" in s or "ssh" in s:
            return 3
        return 4

    y = df[label_col].apply(map_label).astype(np.int32).values

    X = df.select_dtypes(include=[np.number]).values
    X = np.nan_to_num(X, nan=0.0, posinf=1e9, neginf=-1e9)
    X = np.clip(X, -1e9, 1e9).astype(np.float32)

    if X.shape[1] < target_dim:
        X = np.hstack(
            [X, np.zeros((X.shape[0], target_dim - X.shape[1]), dtype=np.float32)]
        )
    else:
        X = X[:, :target_dim]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr.astype(np.float64)).astype(np.float32)
    X_te = scaler.transform(X_te.astype(np.float64)).astype(np.float32)

    _CACHE[key] = ((X_tr, y_tr), (X_te, y_te))
    print(f"✓ Loaded CICIDS2017: Train={X_tr.shape}, Test={X_te.shape}")

    return _CACHE[key]


def get_feature_dim(_=None):
    return 79