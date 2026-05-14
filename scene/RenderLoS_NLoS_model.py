import torch
import torch.nn as nn
import os


class LOSWeightGate(nn.Module):  # v9 v11 v12

    def __init__(self, d_ref=10.0):
        super().__init__()

        self.register_buffer("d_ref", torch.tensor(d_ref))


        self.dist_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
            nn.Softplus()
        )

        self.k = nn.Parameter(torch.tensor(5.0))
        self.alpha = nn.Parameter(torch.tensor(0.7))

    def forward(self, LOS_dist, incident_visibility):

        d2_norm = (LOS_dist / self.d_ref) ** 2  # O(1)


        w_dist = 1.0 / (d2_norm + 1e-6)


        w_corr = self.dist_mlp(d2_norm)


        v_soft = torch.sigmoid(self.k * (incident_visibility - 0.5))
        alpha = torch.sigmoid(self.alpha)
        w_vis = alpha * v_soft + (1 - alpha)

        return w_dist * w_corr * w_vis


class Renderer_LoS_NLoS:

    def __init__(self, tx_power_init=1e3):
        self.tx_power = nn.Parameter(
            torch.tensor(tx_power_init, dtype=torch.float32, device="cuda")
        )

        self.los_gate = LOSWeightGate().to("cuda")

        self.optimizer = None

    def train_setting(self, lr=1e-3):
        params = [
            {'params': [self.tx_power], 'lr': lr, "name": "tx_power"},
            {'params': self.los_gate.parameters(), 'lr': lr, "name": "los_gate"}
        ]
        self.optimizer = torch.optim.Adam(params)

    def compute_weight(self, LOS_dist, incident_visibility):
        return self.los_gate(LOS_dist, incident_visibility)

    def save_weights(self, save_path, iteration=None):
        os.makedirs(save_path, exist_ok=True)

        save_dict = {
            'tx_power': self.tx_power.detach().cpu(),
            'los_gate': self.los_gate.state_dict()
        }

        filename = (
            f"los_nlos_weights_{iteration}.pth"
            if iteration is not None
            else "los_nlos_weights.pth"
        )

        torch.save(save_dict, os.path.join(save_path, filename))
        print(f"[Renderer_LoS_NLoS] Saved weights to {filename}")

    def load_weights(self, load_path):
        if not os.path.exists(load_path):
            print(f"[Renderer_LoS_NLoS] Warning: {load_path} not found.")
            return

        checkpoint = torch.load(load_path, map_location="cuda")

        self.tx_power.data = checkpoint['tx_power'].to("cuda")
        self.los_gate.load_state_dict(checkpoint['los_gate'])

        print(f"[Renderer_LoS_NLoS] Loaded weights from {load_path}")










