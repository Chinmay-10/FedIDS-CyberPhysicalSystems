import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = "experiments/results"
FEDAVG_DIR = os.path.join(BASE_DIR, "fedavg")
PFL_DIR = os.path.join(BASE_DIR, "pfl")
GRAPH_DIR = os.path.join(BASE_DIR, "graphs")

os.makedirs(GRAPH_DIR, exist_ok=True)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def collect_results(folder):
    results = []
    for path in sorted(glob.glob(os.path.join(folder, "*.json"))):
        data = load_json(path)
        results.append(data)
    return results


def plot_privacy_utility(fedavg, pfl):
    def extract(results, method, dist):
        eps, nsl, cic = [], [], []
        for r in results:
            if r["method"] != method or r["data_distribution"] != dist:
                continue
            eps.append(r["epsilon"])
            nsl.append(r["final"]["nsl"]["acc"])
            cic.append(r["final"]["cicids"]["acc"])
        idx = np.argsort(eps)
        return np.array(eps)[idx], np.array(nsl)[idx], np.array(cic)[idx]

    plt.figure(figsize=(9, 6))

    for dist, style in [("iid", "o"), ("noniid", "s")]:
        fe, fn, fc = extract(fedavg, "fedavg", dist)
        pe, pn, pc = extract(pfl, "personalized_fl", dist)

        if len(fe):
            plt.plot(fe, fn, style + "-", label=f"FedAvg NSL ({dist.upper()})")
            plt.plot(fe, fc, style + "--", label=f"FedAvg CICIDS ({dist.upper()})")

        if len(pe):
            plt.plot(pe, pn, style + "-.", label=f"PFL NSL ({dist.upper()})")
            plt.plot(pe, pc, style + ":", label=f"PFL CICIDS ({dist.upper()})")

    plt.xlabel("Privacy Budget ε")
    plt.ylabel("Accuracy")
    plt.title("Privacy–Utility Tradeoff")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    out = os.path.join(GRAPH_DIR, "privacy_utility_tradeoff.png")
    plt.savefig(out, dpi=400)
    plt.close()
    print("Saved:", out)


def plot_pfl_fpr_latency(pfl):
    for metric, ylabel, fname in [
        ("fpr_macro", "False Positive Rate", "pfl_fpr_vs_epsilon.png"),
        ("latency_ms", "Median Latency (ms)", "pfl_latency_vs_epsilon.png"),
    ]:
        plt.figure(figsize=(9, 6))
        for dist, style in [("iid", "o"), ("noniid", "s")]:
            eps, vals = [], []
            for r in pfl:
                if r["data_distribution"] != dist:
                    continue
                eps.append(r["epsilon"])
                vals.append(r["final"]["nsl"][metric])
            idx = np.argsort(eps)
            plt.plot(
                np.array(eps)[idx],
                np.array(vals)[idx],
                style + "-",
                label=f"PFL ({dist.upper()})",
            )

        plt.xlabel("Privacy Budget ε")
        plt.ylabel(ylabel)
        plt.title(f"PFL {ylabel} vs Privacy")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()

        out = os.path.join(GRAPH_DIR, fname)
        plt.savefig(out, dpi=400)
        plt.close()
        print("Saved:", out)

def plot_pfl_communication(pfl):
    plt.figure(figsize=(9, 6))

    for dist, style in [("iid", "o"), ("noniid", "s")]:
        eps, comm = [], []
        for r in pfl:
            if r["data_distribution"] != dist:
                continue
            eps.append(r["epsilon"])
            comm.append(r["final"]["avg_bytes"])
        idx = np.argsort(eps)
        plt.plot(
            np.array(eps)[idx],
            np.array(comm)[idx],
            style + "-",
            label=f"PFL ({dist.upper()})",
        )

    plt.xlabel("Privacy Budget ε")
    plt.ylabel("Average Bytes per Client")
    plt.title("PFL Communication Cost vs Privacy")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    out = os.path.join(GRAPH_DIR, "pfl_comm_vs_epsilon.png")
    plt.savefig(out, dpi=400)
    plt.close()
    print("Saved:", out)

def plot_per_round(results, method):
    for dist in ["iid", "noniid"]:
        for r in results:
            if r["method"] != method or r["data_distribution"] != dist:
                continue

            rounds = r["per_round"]
            xs = [x["round"] for x in rounds]
            nsl = [x["nsl_acc"] for x in rounds]
            cic = [x["cicids_acc"] for x in rounds]

            plt.figure(figsize=(9, 6))
            plt.plot(xs, nsl, label="NSL-KDD")
            plt.plot(xs, cic, label="CICIDS2017")

            plt.xlabel("Federated Round")
            plt.ylabel("Accuracy")
            plt.title(f"{method.upper()} Convergence ({dist.upper()}, ε={r['epsilon']})")
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.legend()
            plt.tight_layout()

            out = os.path.join(
                GRAPH_DIR,
                f"{method}_{dist}_per_round_eps{r['epsilon']}.png",
            )
            plt.savefig(out, dpi=400)
            plt.close()
            print("Saved:", out)


def main():
    fedavg = collect_results(FEDAVG_DIR)
    pfl = collect_results(PFL_DIR)

    plot_privacy_utility(fedavg, pfl)
    plot_pfl_fpr_latency(pfl)
    plot_pfl_communication(pfl)
    plot_per_round(fedavg, "fedavg")
    plot_per_round(pfl, "personalized_fl")

    print("\n✓ All graphs generated in:", GRAPH_DIR)


if __name__ == "__main__":
    main()