import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================
# ORIGINAL RESIDUAL MODEL (for main.py)
# ==========================

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out))
        return F.relu(out + identity)


class FedIDSModel(nn.Module):
    def __init__(self, input_dim=79, num_classes=5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.LayerNorm(256)


        self.res1 = ResidualBlock(256)
        self.res2 = ResidualBlock(256)

        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)

        self.dropout = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.res1(x)
        x = self.res2(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        return self.fc3(x)

# ==============================
# Personalized FL Encoders (UPDATED)
# ==============================

class NSLEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.ln1 = nn.LayerNorm(256)   # FIXED
        self.fc2 = nn.Linear(256, latent_dim)

    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        return self.fc2(x)


class CICIDSEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.ln1 = nn.LayerNorm(512)   # FIXED
        self.fc2 = nn.Linear(512, latent_dim)

    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        return self.fc2(x)



class SharedAlignment(nn.Module):
    def __init__(self, shared_dim=128):
        super().__init__()
        self.fc = nn.Linear(shared_dim, shared_dim)
        self.bn = nn.BatchNorm1d(shared_dim)

    def forward(self, x):
        return F.relu(self.bn(self.fc(x)))


class SharedClassifier(nn.Module):
    def __init__(self, shared_dim=128, num_classes=5):
        super().__init__()
        self.fc = nn.Linear(shared_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


class FederatedModel(nn.Module):
    def __init__(self, encoder, shared_alignment, classifier):
        super().__init__()
        self.encoder = encoder
        self.shared_alignment = shared_alignment
        self.classifier = classifier

    def forward(self, x):
        z = self.encoder(x)
        z = self.shared_alignment(z)
        return self.classifier(z)

    # ===============================
    # FIXED SHARED–STATE EXPORT
    # ===============================
    def get_shared_state(self):
        state = {}

        # flatten alignment params
        for k, v in self.shared_alignment.state_dict().items():
            state[f"shared_alignment.{k}"] = v.clone()

        # flatten classifier params
        for k, v in self.classifier.state_dict().items():
            state[f"classifier.{k}"] = v.clone()

        return state

    # ===============================
    # FIXED SHARED–STATE IMPORT
    # ===============================
    def load_shared_state(self, shared_state):
        own = self.state_dict()

        for k, v in shared_state.items():
            if k in own:
                own[k] = v.clone()

        self.load_state_dict(own, strict=False)


def create_federated_model(input_dim, dataset_type, num_classes=5,
                           latent_dim=128, shared_dim=128):

    if dataset_type.lower() == "nsl_kdd":
        encoder = NSLEncoder(input_dim, latent_dim)
    elif dataset_type.lower() == "cicids2017":
        encoder = CICIDSEncoder(input_dim, latent_dim)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    shared_align = SharedAlignment(shared_dim)
    classifier = SharedClassifier(shared_dim, num_classes)

    return FederatedModel(encoder, shared_align, classifier)
