# 🚀 FedIDS: Fair & Privacy-Preserving Federated Intrusion Detection

**Privacy-Preserving Personalized Federated Learning for Robust Intrusion Detection under Non-IID Data**

---

## 📌 Overview

FedIDS is a research-driven Federated Learning framework designed for real-world Network Intrusion Detection Systems (NIDS).

Modern organizations cannot share raw network traffic due to privacy regulations and security risks. At the same time, data distributions across organizations are highly heterogeneous (Non-IID), causing standard federated learning methods to fail — especially for minority or difficult clients.

FedIDS addresses this challenge by combining:

- Federated Averaging (FedAvg) baseline  
- Personalized Federated Learning (PFL)  
- Differential Privacy (DP)  
- Synergistic optimization (Focal Loss + Cosine LR Scheduling)

The framework enables **collaborative learning without data sharing**, while maintaining stability and fairness under heterogeneous and privacy-constrained settings.

---

## 🎯 Key Technical Contributions

### 1️⃣ Fairness-Oriented Personalized Federated Learning
- Robust to severe Non-IID client distributions  
- Client-specific encoders + shared classifier architecture  
- Prevents minority clients from collapsing  

### 2️⃣ Synergistic Optimization Discovery
- Focal Loss alone improves aggregate accuracy but harms worst-case clients  
- Cosine Annealing LR stabilizes optimization  
- Their combination yields significant fairness gains  

### 3️⃣ Major Performance Improvements

On NSL-KDD:

- **+20.49% overall accuracy improvement**  
  (70.75% → 91.24%)

- **+40.03% worst-case client improvement**  
  (18.41% → 58.45%)

- **47% reduction in client variance**

This demonstrates that optimizing for fairness is as important as optimizing for global accuracy.

### 4️⃣ Goldilocks Effect in Personalization

We identify a non-monotonic relationship between personalization strength (λ) and fairness:

- Weak λ → poor worst-case performance  
- Strong λ → poor worst-case performance  
- Optimal λ → dramatic fairness improvement  

This highlights the importance of careful hyperparameter tuning in real-world federated systems.

---

## 🏗 Architecture

The system separates:

- Local client encoders (private)  
- Shared alignment layer  
- Global classifier  
- Federated aggregation via FedAvg  
- Differential Privacy applied to shared parameters only  

This design preserves privacy while allowing collaborative attack pattern learning.

---

## 🧪 Experimental Setup

### Datasets
- NSL-KDD  
- CICIDS2017  

### Learning Paradigms
- FedAvg  
- Personalized FL  

### Data Distributions
- IID  
- Non-IID (label-skewed partitions)  

### Privacy Budgets
- ε = 0  
- ε = 1  
- ε = 2  
- ε = 5  

### Evaluation Metrics
- Accuracy  
- Macro F1-score  
- Worst-case client accuracy  
- Variance across clients  
- False Positive Rate (FPR)  
- Communication cost  
- Inference latency  
- Privacy–Utility tradeoff  

---

## ⚙️ Installation

```bash
pip install -r requirements.txt


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

Baseline Federated Averaging

```bash
>>python main.py --rounds 20 --epsilon 0 --iid
>>python main.py --rounds 20 --epsilon 2 --noniid
---

Personalized Federated Learning

```bash
>>python personalized_fl.py --rounds 20 --epsilon 1 --iid
>>python personalized_fl.py --rounds 20 --epsilon 5 --noniid
---

## 📈 Generate Visualizations

```bash
python generate_graphs.py
```

Graphs are saved in the `graphs/` directory.

---

🔁 Reproducibility

-Dependency versions specified
-Deterministic data partitioning
-Datasets not included due to licensing
-Results correspond to reported experimental findings

📌 Why This Project Matters

-In real-world federated cybersecurity systems:
-Clients do not have balanced data
-Privacy constraints are strict
-Optimizing only for global accuracy creates unfair systems
-FedIDS demonstrates that fairness-aware optimization is essential for scalable, privacy-preserving collaborative intrusion detection.

---

## 📄 License

Academic and research use only.

---

## Contact

**Author:** Chinmay Patil

**Domain:** Federated Learning · Privacy-Preserving ML · Cybersecurity
