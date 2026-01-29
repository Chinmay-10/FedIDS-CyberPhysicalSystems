# privacy.py
import torch
import numpy as np
from typing import Dict


# ============================================================
# DP-SGD style clipping + Gaussian noise
# ============================================================
def dp_clip_and_noise(update: Dict[str, torch.Tensor],
                      clip: float = 1.0,
                      sigma: float = 0.5,
                      device: str = "cpu") -> Dict[str, torch.Tensor]:
    """
    DP-SGD clipping + noise.
    Used for main FedAvg (flat state_dict updates).
    """
    device = torch.device(device)

    # Compute total L2 norm across all parameters
    total_norm = torch.sqrt(
        sum([(p.detach().to(device) ** 2).sum() for p in update.values()])
    )

    scale = min(1.0, float(clip) / (float(total_norm) + 1e-12))

    noisy_update = {}
    for k, v in update.items():
        v = v.to(device)
        clipped = v * scale

        if sigma > 0:
            noise_std = float(sigma) * float(clip)
            noise = torch.normal(
                0.0, noise_std, size=clipped.shape,
                device=device, dtype=clipped.dtype
            )
            noisy_update[k] = clipped + noise
        else:
            noisy_update[k] = clipped

    return noisy_update


# ============================================================
# Differential Privacy Class
# ============================================================
class DifferentialPrivacy:
    """
    General DP utilities for both main FedAvg + Personalized FL.
    Supports:
      • DP-SGD updates (apply_dp)
      • Nested DP noise for personalized FL (add_noise_to_state)
    """

    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5,
                 clip_norm: float = 1.0, device: str = "cpu"):
        self.epsilon = float(epsilon)
        self.delta = float(delta)
        self.clip_norm = float(clip_norm)
        self.device = torch.device(device)
        self.rounds = 0
        self.privacy_spent = 0.0

        # Gaussian noise multiplier
        self.sigma = self._compute_sigma()

        if self.epsilon > 0:
            print(f"🔒 DP Enabled → ε={self.epsilon}, δ={self.delta}, clip={self.clip_norm}")
            print(f"   Noise multiplier σ = {self.sigma:.6f} (std = σ * clip_norm)")
        else:
            print("🔓 DP OFF — no DP noise added")

    # --------------------------------------------------------
    def _compute_sigma(self) -> float:
        if self.epsilon <= 0:
            return 0.0
        return float(np.sqrt(2.0 * np.log(1.25 / self.delta)) / self.epsilon)

    # --------------------------------------------------------
    # Regular FedAvg DP update
    # --------------------------------------------------------
    def apply_dp(self, update_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if not update_dict:
            return {}
        return dp_clip_and_noise(
            update_dict,
            clip=self.clip_norm,
            sigma=self.sigma,
            device=self.device
        )

    # --------------------------------------------------------
    # Personalized FL DP — noise on nested dicts
    # --------------------------------------------------------
    def add_noise_to_state(self, shared_state):

        noise_std = self.sigma * self.clip_norm

        # Case 1: FLAT dict → {"layer.weight": tensor}
        if isinstance(next(iter(shared_state.values())), torch.Tensor):
            noisy = {}
            for k, v in shared_state.items():
                noise = torch.normal(
                    0.0, noise_std, size=v.shape,
                    device=v.device, dtype=v.dtype
                )
                noisy[k] = v + noise
            return noisy

        # Case 2: NESTED dict → {"layer": {"weight":..., "bias":...}}
        noisy_state = {}
        for block_name, block in shared_state.items():
            noisy_block = {}
            for k, v in block.items():
                if isinstance(v, torch.Tensor):
                    noise = torch.normal(
                        0.0, noise_std, size=v.shape,
                        device=v.device, dtype=v.dtype
                    )
                    noisy_block[k] = v + noise
                else:
                    noisy_block[k] = v
            noisy_state[block_name] = noisy_block

        return noisy_state


    # --------------------------------------------------------
    def update_privacy_budget(self) -> float:
        """Simple conservative accountant."""
        self.rounds += 1

        if self.epsilon <= 0:
            self.privacy_spent = 0.0
        else:
            self.privacy_spent = self.epsilon * np.sqrt(
                2.0 * self.rounds * np.log(1.0 / self.delta)
            )
        return self.privacy_spent
