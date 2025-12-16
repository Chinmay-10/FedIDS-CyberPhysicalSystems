import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class FederatedClient:
    def __init__(self, cid, X, y, device='cpu', dataset_type="nsl_kdd", create_model_fn=None):
        self.cid = cid
        self.X = torch.from_numpy(X).float().to(device)
        self.y = torch.from_numpy(y).long().to(device)
        self.device = device
        self.dataset_type = dataset_type

        assert create_model_fn is not None
        self.model = create_model_fn(
            input_dim=X.shape[1],
            dataset_type=dataset_type
        ).to(device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def local_update(self, epochs=1, batch_size=64):
        self.model.train()
        idx = np.arange(len(self.X))

        for _ in range(epochs):
            np.random.shuffle(idx)
            for i in range(0, len(idx), batch_size):
                batch = idx[i:i+batch_size]

                xb = self.X[batch]
                yb = self.y[batch]

                self.optimizer.zero_grad()
                out = self.model(xb)
                loss = self.criterion(out, yb)
                loss.backward()
                self.optimizer.step()

    # Correct: returns flattened shared weights
    def get_shared_state(self):
        return self.model.get_shared_state()

    # Correct: loads only alignment + classifier
    def load_shared_state(self, shared_state):
        self.model.load_shared_state(shared_state)

    def eval_local(self):
        self.model.eval()
        with torch.no_grad():
            pred = self.model(self.X).argmax(1).cpu().numpy()
            return (pred == self.y.cpu().numpy()).mean()
