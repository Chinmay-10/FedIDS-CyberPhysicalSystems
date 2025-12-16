import os
import json
import time
import argparse
from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn

from utils import preprocessing as pp
from utils.partitioning import (
    create_client_shards,
    create_client_shards_single_dataset,
    create_client_shards_noniid,
)
from models.neural_network_torch import FedIDSModel
from metrics import compute_classification_metrics, compute_latency_median_ms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def avg_state_dicts(states: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        k: sum(s[k].float() for s in states) / len(states)
        for k in states[0].keys()
    }

def train_local(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    lr: float,
    epochs: int,
    dp_sigma: float,
):
    import torch.nn.functional as F
    from torch.nn.utils import clip_grad_norm_

    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    for _ in range(epochs):
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(X), y, label_smoothing=0.1)
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    if dp_sigma > 0:
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * dp_sigma)

    return model.state_dict()

def eval_model(model: nn.Module, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    model.to(DEVICE).eval()
    with torch.no_grad():
        preds = model(torch.from_numpy(X).float().to(DEVICE)) \
            .argmax(dim=1).cpu().numpy()
    return compute_classification_metrics(y, preds)

def save_results(
    epsilon: float,
    use_dp: bool,
    args,
    nsl_metrics,
    cic_metrics,
    latency_nsl,
    latency_cic,
    per_round,
    avg_bytes,
):
    out_dir = os.path.join("experiments", "results", "fedavg")
    os.makedirs(out_dir, exist_ok=True)

    data_distribution = "noniid" if args.noniid else "iid"

    out = {
        "method": "fedavg",
        "epsilon": float(epsilon),
        "data_distribution": data_distribution,
        "config": vars(args),
        "final": {
            "nsl": {
                "acc": nsl_metrics["acc"],
                "f1_macro": nsl_metrics["f1_macro"],
                "fpr_macro": nsl_metrics["fpr_macro"],
                "latency_ms": latency_nsl,
            },
            "cicids": {
                "acc": cic_metrics["acc"],
                "f1_macro": cic_metrics["f1_macro"],
                "fpr_macro": cic_metrics["fpr_macro"],
                "latency_ms": latency_cic,
            },
            "avg_bytes": avg_bytes,
        },
        "per_round": per_round,
    }

    path = os.path.join(
        out_dir,
        f"fedavg_{data_distribution}_eps{epsilon:g}.json"
    )

    with open(path, "w") as f:
        json.dump(out, f, indent=4)

    print(f"✓ Saved FedAvg results → {path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["both", "nsl", "cicids"], default="both")
    parser.add_argument("--clients", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--noniid", action="store_true")
    args = parser.parse_args()

    use_dp = args.epsilon > 0
    dp_sigma = (
        np.sqrt(2 * np.log(1.25 / args.delta)) / args.epsilon
        if use_dp else 0.0
    )

    (Xn_tr, yn_tr), (Xn_te, yn_te) = pp.load_nsl_kdd_data()
    (Xc_tr, yc_tr), (Xc_te, yc_te) = pp.load_cicids2017_data(nrows=150000)

    datasets = {
        "nsl": ((Xn_tr, yn_tr), (Xn_te, yn_te)),
        "cicids": ((Xc_tr, yc_tr), (Xc_te, yc_te)),
    }

    if args.dataset == "nsl":
        shards = create_client_shards_single_dataset(
            Xn_tr, yn_tr, args.clients, "data/clients_nsl", "nsl"
        )
    elif args.dataset == "cicids":
        shards = create_client_shards_single_dataset(
            Xc_tr, yc_tr, args.clients, "data/clients_cicids", "cic"
        )
    else:
        shards = (
            create_client_shards_noniid(datasets, args.clients, "data/clients_noniid")
            if args.noniid
            else create_client_shards(datasets, args.clients, "data/clients")
        )

    num_classes = int(max(np.max(yn_tr), np.max(yc_tr)) + 1)
    server = FedIDSModel(pp.get_feature_dim(), num_classes).to(DEVICE)
    server_sd = server.state_dict()

    per_round, bytes_round = [], []

    for r in range(1, args.rounds + 1):
        t0 = time.time()
        client_sds, total_bytes = [], 0

        for Xc, yc in shards:
            loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(
                    torch.from_numpy(Xc).float(),
                    torch.from_numpy(yc).long(),
                ),
                batch_size=args.batch_size,
                shuffle=True,
            )

            local = FedIDSModel(pp.get_feature_dim(), num_classes)
            local.load_state_dict(server_sd, strict=False)

            sd = train_local(local, loader, DEVICE, args.lr, args.local_epochs, dp_sigma)
            client_sds.append(sd)
            total_bytes += sum(p.numel() for p in sd.values()) * 4

        server_sd = avg_state_dicts(client_sds)

        temp_model = FedIDSModel(pp.get_feature_dim(), num_classes)
        temp_model.load_state_dict(server_sd, strict=False)

        nsl_round = eval_model(temp_model, Xn_te, yn_te)
        cic_round = eval_model(temp_model, Xc_te, yc_te)

        avg_bytes = total_bytes / len(shards)
        bytes_round.append(avg_bytes)

        per_round.append({
            "round": r,
            "nsl_acc": nsl_round["acc"],
            "cicids_acc": cic_round["acc"],
            "avg_bytes": avg_bytes,
            "time_s": time.time() - t0,
        })

        print(f"Round {r}/{args.rounds} completed")

    final_model = FedIDSModel(pp.get_feature_dim(), num_classes)
    final_model.load_state_dict(server_sd, strict=False)

    nsl_metrics = eval_model(final_model, Xn_te, yn_te)
    cic_metrics = eval_model(final_model, Xc_te, yc_te)

    latency_nsl = compute_latency_median_ms(final_model, Xn_te, device=DEVICE)
    latency_cic = compute_latency_median_ms(final_model, Xc_te, device=DEVICE)

    save_results(
        args.epsilon,
        use_dp,
        args,
        nsl_metrics,
        cic_metrics,
        latency_nsl,
        latency_cic,
        per_round,
        float(np.mean(bytes_round)),
    )


if __name__ == "__main__":
    main()