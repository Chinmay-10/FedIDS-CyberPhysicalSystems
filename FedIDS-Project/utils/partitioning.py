import os
import numpy as np

def partition_iid(X, y, num_clients):
    idx = np.random.permutation(len(X))
    splits = np.array_split(idx, num_clients)
    return [(X[s], y[s]) for s in splits]


def partition_noniid(X, y, num_clients, alpha=0.5):
    y = np.asarray(y)
    classes = np.unique(y)
    client_indices = [[] for _ in range(num_clients)]

    for cls in classes:
        cls_idx = np.where(y == cls)[0]
        np.random.shuffle(cls_idx)

        proportions = np.random.dirichlet([alpha] * num_clients)
        counts = (proportions * len(cls_idx)).astype(int)
        counts[np.argmax(proportions)] += len(cls_idx) - counts.sum()

        start = 0
        for i in range(num_clients):
            end = start + counts[i]
            client_indices[i].extend(cls_idx[start:end])
            start = end

    shards = []
    feat_dim = X.shape[1]
    for idxs in client_indices:
        if len(idxs) == 0:
            shards.append(
                (np.zeros((0, feat_dim), dtype=X.dtype),
                 np.zeros((0,), dtype=y.dtype))
            )
        else:
            shards.append((X[idxs], y[idxs]))

    return shards

def create_client_shards(datasets, num_clients, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    (Xn, yn), _ = datasets["nsl"]
    (Xc, yc), _ = datasets["cicids"]

    mid = num_clients // 2

    shards_nsl = partition_iid(Xn, yn, mid)
    shards_cic = partition_iid(Xc, yc, num_clients - mid)

    shards = shards_nsl + shards_cic

    for i, (X, y) in enumerate(shards):
        np.save(os.path.join(out_dir, f"client_{i}_X.npy"), X)
        np.save(os.path.join(out_dir, f"client_{i}_y.npy"), y)

    return shards


def create_client_shards_single_dataset(X, y, num_clients, out_dir, prefix):
    os.makedirs(out_dir, exist_ok=True)

    idx = np.random.permutation(len(X))
    splits = np.array_split(idx, num_clients)

    shards = []
    for i, s in enumerate(splits):
        Xc, yc = X[s], y[s]
        np.save(os.path.join(out_dir, f"{prefix}_client_{i}_X.npy"), Xc)
        np.save(os.path.join(out_dir, f"{prefix}_client_{i}_y.npy"), yc)
        shards.append((Xc, yc))

    return shards

def create_heterogeneous_clients(
    Xn, yn,
    Xc, yc,
    num_clients=10,
    noniid=False,
    alpha=0.5,
):

    half = num_clients // 2

    if noniid:
        shards_nsl = partition_noniid(Xn, yn, half, alpha)
        shards_cic = partition_noniid(Xc, yc, num_clients - half, alpha)
    else:
        shards_nsl = partition_iid(Xn, yn, half)
        shards_cic = partition_iid(Xc, yc, num_clients - half)

    return shards_nsl + shards_cic

def create_client_shards_noniid(datasets, num_clients, out_dir, alpha=0.5):
    os.makedirs(out_dir, exist_ok=True)

    (Xn, yn), _ = datasets["nsl"]
    (Xc, yc), _ = datasets["cicids"]

    mid = num_clients // 2

    shards_nsl = partition_noniid(Xn, yn, mid, alpha)
    shards_cic = partition_noniid(Xc, yc, num_clients - mid, alpha)

    shards = shards_nsl + shards_cic

    for i, (X, y) in enumerate(shards):
        np.save(os.path.join(out_dir, f"client_{i}_X.npy"), X)
        np.save(os.path.join(out_dir, f"client_{i}_y.npy"), y)

    return shards
