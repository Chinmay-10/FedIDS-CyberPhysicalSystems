import torch
import numpy as np
import copy

class FederatedServer:

    def __init__(self, global_model, device, test_loader,
                 num_rounds=20):
        self.global_model = global_model
        self.device = device
        self.test_loader = test_loader
        self.num_rounds = num_rounds

    # ----------------------------------------------------------
    # AVERAGE ONLY CLASSIFIER LAYERS
    # client_states is a list of classifier state_dicts
    # ----------------------------------------------------------
    def fedavg(self, client_states):
        new_state = copy.deepcopy(client_states[0])
        for key in new_state.keys():
            for cs in client_states[1:]:
                new_state[key] = new_state[key] + cs[key]
            new_state[key] = new_state[key] / len(client_states)
        return new_state

    # ----------------------------------------------------------
    # MAIN FL LOOP
    # ----------------------------------------------------------
    def train_federated(self, clients):
        comm_stats = []
        for r in range(self.num_rounds):
            print(f"\n=== Federated Round {r+1}/{self.num_rounds} ===")

            client_updates = []
            comm_bytes_list = []

            # 1. Local training (clients return (state_dict, bytes))
            for c in clients:
                update, bytes_sent = c.local_train()
                client_updates.append(update)
                comm_bytes_list.append(bytes_sent)

            # 2. Average classifier weights
            new_classifier_state = self.fedavg(client_updates)

            # 3. Update global model
            self.global_model.load_classifier_state(new_classifier_state)

            # 4. Broadcast new classifier to clients
            for c in clients:
                c.model.load_classifier_state(new_classifier_state)

            # 5. Evaluate global model (simple accuracy)
            acc = self.evaluate_global()
            avg_bytes = float(np.mean(comm_bytes_list)) if comm_bytes_list else 0.0
            total_bytes = float(np.sum(comm_bytes_list)) if comm_bytes_list else 0.0
            comm_stats.append({'round': r+1, 'avg_bytes': avg_bytes, 'total_bytes': total_bytes})

            print(f" Global Test Accuracy: {acc:.4f} | Avg bytes/client: {avg_bytes:.0f} | Total bytes: {total_bytes:.0f}")

        return {
            'comm_stats': comm_stats
        }

    # ----------------------------------------------------------
    # EVALUATE GLOBAL CLASSIFIER + LOCAL ENCODERS
    # ----------------------------------------------------------
    def evaluate_global(self):
        self.global_model.eval()

        preds = []
        true = []

        with torch.no_grad():
            for xb, yb in self.test_loader:
                xb = xb.to(self.device)
                out = self.global_model(xb)
                preds.extend(out.argmax(1).cpu().numpy())
                true.extend(yb.numpy())

        from sklearn.metrics import accuracy_score
        return accuracy_score(true, preds)