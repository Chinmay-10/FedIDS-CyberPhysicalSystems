# FedIDS
## Privacy-Preserving Personalized Federated Learning for Intrusion Detection Systems

## 📌 Overview

**FedIDS** is a research-oriented Federated Learning (FL) framework designed for **Network Intrusion Detection Systems (NIDS)** that enables collaborative model training across multiple clients **without sharing raw network traffic data**.

The framework combines:
- **Federated Averaging (FedAvg)** as a baseline
- **Personalized Federated Learning (PFL)** for handling data heterogeneity
- **Differential Privacy (DP)** for formal privacy guarantees

FedIDS specifically addresses two critical real-world challenges in cybersecurity:
1. **Non-IID data distributions across organizations**
2. **Strict privacy requirements on sensitive network traffic**

This repository contains the **complete, frozen, and reproducible implementation** used for experimental evaluation and research publication.

---

## 🎯 Key Contributions

- Personalized Federated Learning architecture robust to **Non-IID client data**
- Differential Privacy with configurable **privacy budget (ε)**
- Extensive evaluation on **NSL-KDD** and **CICIDS2017** datasets
- Privacy–utility tradeoff analysis
- Communication-efficient training (~70 KB per client per round in PFL)
- Stable convergence under heterogeneous and privacy-constrained settings

---

## 🏗️ Project Structure

```

FedIDS-Project/
├── main.py
├── personalized_fl.py
├── federated_server.py
├── federated_client.py
├── preprocessing.py
├── privacy.py
├── metrics.py
├── generate_graphs.py
├── results/
├── graphs/
├── requirements.txt
├── .gitignore
└── README.md

---

## 🧪 Experimental Design

### Learning Paradigms
- Federated Averaging (FedAvg)
- Personalized Federated Learning (PFL)

### Data Distributions
- IID
- Non-IID (label-skewed client partitions)

### Privacy Budgets
- ε = 0 (No Differential Privacy)
- ε = 1
- ε = 2
- ε = 5

### Evaluation Metrics
- Accuracy
- Macro F1-score
- False Positive Rate (FPR)
- Convergence Stability
- Communication Cost
- Median Inference Latency
- Privacy–Utility Tradeoff

---

## 📊 Key Findings

- FedAvg performs well only under IID and non-private settings
- FedAvg becomes unstable under Non-IID data and strong privacy constraints
- PFL maintains stable convergence under Non-IID data
- PFL exhibits a more graceful privacy–utility tradeoff
- Communication overhead in PFL remains nearly constant across ε values
- Differential Privacy reduces false positives by acting as regularization

---

## ⚙️ Installation

```bash
pip install -r requirements.txt

---

## ▶️ Running Experiments

```bash
python main.py --rounds 20 --epsilon 0 --iid
python main.py --rounds 20 --epsilon 2 --noniid
python personalized_fl.py --rounds 20 --epsilon 1 --iid
python personalized_fl.py --rounds 20 --epsilon 5 --noniid
```

---

## 📈 Generate Figures

```bash
python generate_graphs.py
```

Graphs are saved in the `graphs/` directory.

---

## 🔁 Reproducibility Notes

* Dependency versions are locked
* Datasets are not included due to licensing
* Results correspond exactly to the paper experiments
* Repository is version-tagged for reproducibility

---

## 📄 License

Academic and research use only.

---

## Contact

**Author:** Chinmay Patil
**Domain:** Federated Learning · Privacy-Preserving ML · Cybersecurity

