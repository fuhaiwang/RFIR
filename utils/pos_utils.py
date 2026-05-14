import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils.rigid_utils import exp_se3


def get_embedder(multires, i=1):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': i,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)



class FreqCondNetwork(nn.Module): # v47
    def __init__(self, xyz_dim=3, freq_dim=6892, hidden_dim=256, cond_dim=128):
        super().__init__()
        from utils.pos_utils import get_embedder

        self.embed_fn_pos, out_dim_pos = get_embedder(multires=10, i=xyz_dim)

        self.L_f = 4
        self.hidden_dim = hidden_dim
        self.cond_dim = cond_dim

        self.f_encoder = nn.Sequential(
            nn.Linear(2 * self.L_f, 64),
            nn.SiLU(),
            nn.Linear(64, cond_dim),
            nn.SiLU()
        )

        self.freq_proj = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.num_layers = 6
        self.layers = nn.ModuleList()
        self.acts = nn.ModuleList()

        for i in range(self.num_layers):
            if i == 0:
                in_dim = out_dim_pos + hidden_dim   # <<< concat freq
            else:
                in_dim = hidden_dim

            self.layers.append(nn.Linear(in_dim, hidden_dim))
            self.acts.append(nn.SiLU(inplace=True))

        self.film_generators = nn.ModuleList([
            nn.Linear(cond_dim, hidden_dim * 2)
            for _ in range(self.num_layers)
        ])


        self.final = nn.Linear(hidden_dim, 3)

        nn.init.zeros_(self.final.weight)
        nn.init.zeros_(self.final.bias)

        self.freq_dim = freq_dim

    def forward(self, x, f):

        if f.dim() == 2 and f.shape[0] > 1:
            f_input = f[:1]
        else:
            f_input = f

        f_sq = f_input.reshape(-1, 1)
        f_center, f_bw = 5.8e9, 160e6
        f_norm = (f_sq - f_center) / (f_bw / 2)

        freq_bands = 2.0 ** torch.arange(self.L_f, device=f_norm.device)
        f_norm_multi = f_norm * math.pi * freq_bands
        f_pe = torch.cat([torch.sin(f_norm_multi), torch.cos(f_norm_multi)], dim=-1)

        f_emb = self.f_encoder(f_pe)          # [B, cond_dim]
        f_cond = self.freq_proj(f_emb)        # [B, hidden_dim]

        h = self.embed_fn_pos(x)               # [N, out_dim_pos]

        # broadcast freq conditioning
        f_cond = f_cond.unsqueeze(0).expand(h.shape[0], -1, -1)  # [N, B, hidden_dim]
        h = h.unsqueeze(1).expand(-1, f_cond.shape[1], -1)       # [N, B, out_dim_pos]


        h = torch.cat([h, f_cond], dim=-1)     # [N, B, out_dim_pos + hidden_dim]

        for i, (layer, act, film_gen) in enumerate(zip(self.layers, self.acts, self.film_generators)):

            if i == 0:
                h = layer(h)
                h = act(h)
            else:
                h_res = h

                h = layer(h)
                h = act(h)

                # 弱 FiLM
                gamma_beta = film_gen(f_emb)       # [B, 2H]
                gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)

                gamma = gamma.unsqueeze(0)
                beta = beta.unsqueeze(0)

                h = h + 0.05 * (gamma * h + beta)
                h = h + h_res

        out = self.final(h)                    # [N, B, 4]
        out = out.permute(2, 0, 1)             # [4, N, B]

        d_reflection_coe   = bounded_delta(out[0], 20.0)
        d_reflection_phase = bounded_delta(out[1], math.pi)
        d_roughness        = bounded_delta(out[2], 5.0)


        return d_reflection_coe, d_reflection_phase, d_roughness

def bounded_delta(x, scale):
    return scale * x / (1.0 + x.abs())




