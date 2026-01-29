import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
#  NSL-KDD Encoder (input_dim auto-detected from preprocessing)
# ======================================================================
class NSLEncoder(nn.Module):
    def __init__(self, input_dim=41, latent_dim=128):
        super(NSLEncoder, self).__init__()

        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.ln1 = nn.LayerNorm(128)

        self.fc2 = nn.Linear(128, latent_dim)
        self.bn2 = nn.BatchNorm1d(latent_dim)
        self.ln2 = nn.LayerNorm(latent_dim)

    def forward(self, x):
        # (batch, features)
        x = F.relu(self.ln1(self.bn1(self.fc1(x))))
        x = F.relu(self.ln2(self.bn2(self.fc2(x))))
        return x  # latent vector


# ======================================================================
#  CICIDS2017 Encoder (input_dim usually ~78–85 depending on CSV features)
# ======================================================================
class CICIDSEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=128):
        super(CICIDSEncoder, self).__init__()

        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.ln1 = nn.LayerNorm(256)

        self.fc2 = nn.Linear(256, latent_dim)
        self.bn2 = nn.BatchNorm1d(latent_dim)
        self.ln2 = nn.LayerNorm(latent_dim)

    def forward(self, x):
        x = F.relu(self.ln1(self.bn1(self.fc1(x))))
        x = F.relu(self.ln2(self.bn2(self.fc2(x))))
        return x


# ======================================================================
#  Shared Global Classifier (FedAvg runs on this module ONLY)
# ======================================================================
class SharedClassifier(nn.Module):
    def __init__(self, latent_dim=128, num_classes=5):
        super(SharedClassifier, self).__init__()

        self.fc1 = nn.Linear(latent_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.ln1 = nn.LayerNorm(64)

        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.ln1(self.bn1(self.fc1(x))))
        x = self.fc2(x)
        return x


# ======================================================================
#  Federated Wrapper — Combines Local Encoder + Shared Classifier
#  (Only classifier is shared; encoder stays local to each client)
# ======================================================================
class FederatedIDSModel(nn.Module):
    def __init__(self, encoder, classifier):
        super(FederatedIDSModel, self).__init__()
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, x):
        latent = self.encoder(x)
        logits = self.classifier(latent)
        return logits

    # -------------------------------------------------------------
    # Federated Weight Sharing (classifier only)
    # -------------------------------------------------------------
    def get_classifier_state(self):
        """
        Return classifier weights on CPU for FedAvg.
        (client → server)
        """
        state = {
            k: v.clone().detach().cpu().float()
            for k, v in self.classifier.state_dict().items()
        }
        return state

    def load_classifier_state(self, new_state):
        """
        Load federated classifier weights.
        (server → client)
        """
        # Ensure float32 consistency
        clean_state = {k: v.float() for k, v in new_state.items()}
        self.classifier.load_state_dict(clean_state, strict=False)
