import os
import json
import time
import argparse
from typing import List, Dict

import numpy as np
import torch

from utils import preprocessing as pp
from utils.partitioning import create_heterogeneous_clients
from federated_client import FederatedClient
from models.neural_network_torch import create_federated_model
from models.privacy import DifferentialPrivacy
from metrics import compute_classification_metrics, compute_latency_median_ms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def fedavg_shared_states(states: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        k: sum(s[k].float() for s in states) / len(states)
        for k in states[0].keys()
    }


def apply_dp(state: Dict[str, torch.Tensor], dp: DifferentialPrivacy | None):
    return dp.apply_dp(state) if dp is not None else state

def save_results(
    epsilon,
    use_privacy,
    dp,
    args,
    nsl_metrics,
    cic_metrics,
    latency_nsl,
    latency_cic,
    per_round,
    avg_bytes,
):
    out_dir = os.path.join("experiments", "results", "pfl")
    os.makedirs(out_dir, exist_ok=True)

    data_distribution = "noniid" if args.noniid else "iid"

    dp_info = {
        "enabled": bool(use_privacy),
        "epsilon": float(dp.epsilon) if dp else float(epsilon),
        "delta": float(dp.delta) if dp else float(args.delta),
        "clip_norm": float(dp.clip_norm) if dp else float(args.clip),
        "sigma": float(dp.sigma) if dp else 0.0,
        "privacy_spent": float(dp.privacy_spent) if dp else 0.0,
    }

    out = {
        "method": "personalized_fl",
        "epsilon": float(epsilon),
        "data_distribution": data_distribution,
        "dp": dp_info,
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
        f"pfl_{data_distribution}_eps{epsilon:g}.json"
    )

    with open(path, "w") as f:
        json.dump(out, f, indent=4)

    print(f"✓ Saved PFL results → {path}")

def build_server_models(shared_state, input_dim, num_classes):
    from models.neural_network_torch import (
        NSLEncoder,
        CICIDSEncoder,
        SharedAlignment,
        SharedClassifier,
        FederatedModel,
    )

    align = SharedAlignment(shared_dim=128)
    clf = SharedClassifier(shared_dim=128, num_classes=num_classes)

    nsl_model = FederatedModel(NSLEncoder(input_dim), align, clf)
    cic_model = FederatedModel(CICIDSEncoder(input_dim), align, clf)

    nsl_model.load_shared_state(shared_state)
    cic_model.load_shared_state(shared_state)

    return nsl_model, cic_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--clients", type=int, default=10)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--noniid", action="store_true")
    parser.add_argument("--use_privacy", action="store_true")
    args = parser.parse_args()

    use_privacy = args.use_privacy and args.epsilon > 0

    dp = (
        DifferentialPrivacy(
            epsilon=args.epsilon,
            delta=args.delta,
            clip_norm=args.clip,
            device=str(DEVICE),
        )
        if use_privacy
        else None
    )

    (Xn_tr, yn_tr), (Xn_te, yn_te) = pp.load_nsl_kdd_data()
    (Xc_tr, yc_tr), (Xc_te, yc_te) = pp.load_cicids2017_data()

    client_data = create_heterogeneous_clients(
        Xn_tr, yn_tr, Xc_tr, yc_tr,
        num_clients=args.clients,
        noniid=args.noniid,
    )

    input_dim = Xn_tr.shape[1]
    num_classes = int(max(np.max(yn_tr), np.max(yc_tr)) + 1)

    clients = []
    half = len(client_data) // 2

    for cid, (X, y) in enumerate(client_data):
        dtype = "nsl_kdd" if cid < half else "cicids2017"
        clients.append(
            FederatedClient(
                cid=cid,
                X=X,
                y=y,
                device=DEVICE,
                dataset_type=dtype,
                create_model_fn=lambda **kw: create_federated_model(
                    kw["input_dim"],
                    kw["dataset_type"],
                    num_classes,
                ),
            )
        )

    shared_state = None
    per_round = []
    bytes_round = []

    for r in range(1, args.rounds + 1):
        t0 = time.time()
        states, total_bytes = [], 0
        nsl_acc, cic_acc = [], []

        for c in clients:
            if shared_state is not None:
                c.load_shared_state(shared_state)

            c.local_update(args.local_epochs, args.batch_size)
            s = apply_dp(c.get_shared_state(), dp)
            states.append(s)

            acc = c.eval_local()
            (nsl_acc if c.dataset_type == "nsl_kdd" else cic_acc).append(acc)

            total_bytes += sum(t.numel() for t in s.values()) * 4

        shared_state = fedavg_shared_states(states)
        eps_spent = dp.update_privacy_budget() if dp else None

        avg_bytes = total_bytes / len(clients)
        bytes_round.append(avg_bytes)

        per_round.append({
            "round": r,
            "nsl_acc": float(np.mean(nsl_acc)) if nsl_acc else 0.0,
            "cicids_acc": float(np.mean(cic_acc)) if cic_acc else 0.0,
            "avg_bytes": avg_bytes,
            "time_s": time.time() - t0,
            "eps_spent": eps_spent,
        })

        print(f"Round {r}/{args.rounds} completed")

    nsl_model, cic_model = build_server_models(shared_state, input_dim, num_classes)

    with torch.no_grad():
        preds_n = nsl_model(torch.tensor(Xn_te).float().to(DEVICE)) \
            .argmax(1).cpu().numpy()
        preds_c = cic_model(torch.tensor(Xc_te).float().to(DEVICE)) \
            .argmax(1).cpu().numpy()

    nsl_metrics = compute_classification_metrics(yn_te, preds_n)
    cic_metrics = compute_classification_metrics(yc_te, preds_c)

    latency_nsl = compute_latency_median_ms(nsl_model, Xn_te, device=DEVICE)
    latency_cic = compute_latency_median_ms(cic_model, Xc_te, device=DEVICE)

    save_results(
        args.epsilon,
        use_privacy,
        dp,
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