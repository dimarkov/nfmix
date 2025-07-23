import argparse
import time
import gc
import math

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms
from einops import rearrange, repeat
import matplotlib.pyplot as plt

import torch_optimizer
import ivon
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import zuko

# helpers


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


# helper classes from settrans_noprop


class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)
        if exists(self.norm_context):
            context = kwargs["context"]
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)
        return self.fn(x, **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2), GEGLU(), nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head**-0.5
        self.heads = heads
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context=None, mask=None):
        h = self.heads
        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))
        sim = einsum("b i d, b j d -> b i j", q, k) * self.scale
        if exists(mask):
            mask = rearrange(mask, "b ... -> b (...)")
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, "b j -> (b h) () j", h=h)
            sim.masked_fill_(~mask, max_neg_value)
        attn = sim.softmax(dim=-1)
        out = einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)


# Set Transformer Components


class MAB(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head)
        self.ff = FeedForward(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, y):
        x = self.norm1(self.attn(x, context=y) + x)
        x = self.norm2(self.ff(x) + x)
        return x


class SAB(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        self.mab = MAB(dim, heads=heads, dim_head=dim_head)

    def forward(self, x):
        return self.mab(x, x)


class ISAB(nn.Module):
    def __init__(self, dim, num_classes, heads=8, dim_head=64, num_inducing_points=16):
        super().__init__()
        self.inducing_points = nn.Parameter(
            torch.randn(num_classes, num_inducing_points, dim)
        )
        self.mab1 = MAB(dim, heads=heads, dim_head=dim_head)
        self.mab2 = MAB(dim, heads=heads, dim_head=dim_head)

    def forward(self, x, y):
        h = self.mab1(self.inducing_points[y], x)
        return self.mab2(x, h)


class PMA(nn.Module):
    def __init__(self, dim, num_classes, heads=8, dim_head=64, num_seeds=1):
        super().__init__()
        self.seeds = nn.Parameter(torch.randn(num_classes, num_seeds, dim))
        self.mab = MAB(dim, heads=heads, dim_head=dim_head)
        self.ff = FeedForward(dim)

    def forward(self, x, y):
        x = self.ff(x)
        return self.mab(self.seeds[y], x)


class SetTransformer(nn.Module):
    def __init__(
        self,
        *,
        num_classes,
        dim,
        depth,
        heads,
        dim_head,
        num_inducing_points,
        num_seeds=1,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            *[
                ISAB(
                    dim,
                    num_classes,
                    heads=heads,
                    dim_head=dim_head,
                    num_inducing_points=num_inducing_points,
                )
                for _ in range(depth)
            ]
        )
        self.decoder = nn.Sequential(
            PMA(dim, num_classes, heads=heads, dim_head=dim_head, num_seeds=num_seeds),
            SAB(dim, heads=heads, dim_head=dim_head),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.LayerNorm(dim),
        )

    def forward(self, x, y):
        for enc in self.encoder:
            x = enc(x, y)

        x = self.decoder[0](x, y)
        for dec in self.decoder[1:]:
            x = dec(x)
        return x


def optim_sched(model, datasize, hess_init, weight_decay, num_epochs):
    optimizer = ivon.IVON(
        model.parameters(),
        lr=1e-2,
        weight_decay=weight_decay,
        hess_init=hess_init,
        ess=datasize,
    )
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_epochs=num_epochs // 5,
        max_epochs=num_epochs,
        warmup_start_lr=1e-3,
        eta_min=1e-4,
    )
    return optimizer, scheduler


def evaluate_acc(model, x, y, num_classes, device, num_samples=1, η: float = 1.0):
    elbos = []
    for c in torch.arange(num_classes, device=device):
        elbo = 0.0
        for _ in range(num_samples):
            elbo -= compute_loss(
                model, x, c.repeat(x.size(0)), device, η, add_noise=False
            ).detach()
        elbos.append(elbo / num_samples)
    pred = torch.stack(elbos).argmax(dim=0)
    num_correct = (pred == y).sum().item()
    return num_correct


# ----------------------------------------------------------------------------
# Sinusoidal embedding for scalar t ∈ [0,1]
# ----------------------------------------------------------------------------
def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000)
        * torch.arange(half, device=t.device, dtype=t.dtype)
        / (half - 1)
    )
    args = t * freqs.unsqueeze(0)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


# ----------------------------------------------------------------------------


class TimeEncoder(nn.Module):
    def __init__(self, time_emb_dim: int, embed_dim: int):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(time_emb_dim, embed_dim), nn.ReLU())
        self.time_emb_dim = time_emb_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        te = sinusoidal_embedding(t, self.time_emb_dim)
        return self.fc(te)


class NoiseSchedule(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.gamma_tilde = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )
        self.gamma0 = nn.Parameter(torch.tensor(-7.0))
        self.gamma1 = nn.Parameter(torch.tensor(7.0))

    def _gamma_bar(self, t: torch.Tensor) -> torch.Tensor:
        g0 = self.gamma_tilde(torch.zeros_like(t))
        g1 = self.gamma_tilde(torch.ones_like(t))
        return ((self.gamma_tilde(t) - g0) / (g1 - g0 + 1e-8)).clamp(0, 1)

    def alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        γt = self.gamma0 + (self.gamma1 - self.gamma0) * (1 - self._gamma_bar(t))
        return torch.sigmoid(-γt / 2).clamp(1e-5, 1 - 1e-5)


class GaussianPositionalEncoding(torch.nn.Module):
    def __init__(self, channels, num_freqs=16, sigma=1.0):
        super().__init__()
        self.num_freqs = num_freqs
        self.b = torch.nn.Parameter(torch.randn((num_freqs, 2)) * sigma)
        fourier_dim = 2 * num_freqs
        self.proj = torch.nn.Linear(fourier_dim, channels)

    def forward(self, coords):
        coords_freqs = (2 * torch.pi * coords) @ self.b.T
        features = torch.cat([torch.sin(coords_freqs), torch.cos(coords_freqs)], dim=-1)
        return self.proj(features)


class ZEncoder(nn.Module):
    """
    Encodes a embedding vector z_t (shape [B, in_dim]) via a small FC net.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 2 * in_dim),
            nn.LayerNorm(2 * in_dim),
            nn.ReLU(),
            nn.Linear(2 * in_dim, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# ----------------------------------------------------------------------------
# fuse head to combine image, z, and t features
# ----------------------------------------------------------------------------
class FuseHead(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(
        self, fy: torch.Tensor, fz: torch.Tensor, ft: torch.Tensor
    ) -> torch.Tensor:
        x = torch.cat([fy, fz, ft], dim=-1)
        return self.net(x.view(x.size(0), -1))


# Main Model: NoPropSetTransformerFlow
class NoPropSetTransformerFlow(nn.Module):
    def __init__(
        self,
        dim,
        num_classes,
        embed_dim=256,
        time_emb_dim=64,
        set_trans_depth=2,
        set_trans_heads=8,
        set_trans_dim_head=64,
        set_trans_inducing_points=4,
        flow_type="maf",
        flow_transforms=1,
        flow_dim=None,
        num_seeds=2,
        mixture_components=1,
    ):
        super().__init__()
        self.dim = dim
        self.z_shape = (num_seeds, embed_dim)

        self.positional_encoding = GaussianPositionalEncoding(dim)

        self.embed_data = nn.Linear(2 * dim, embed_dim)

        self.set_trans = SetTransformer(
            num_classes=num_classes,
            dim=embed_dim,
            depth=set_trans_depth,
            heads=set_trans_heads,
            dim_head=set_trans_dim_head,
            num_inducing_points=set_trans_inducing_points,
            num_seeds=num_seeds,
        )

        self.time_enc = TimeEncoder(time_emb_dim, embed_dim)
        self.noise_schedule = NoiseSchedule(hidden_dim=64)

        z_dim = 2 * embed_dim * num_seeds
        self.z_encoder = ZEncoder(z_dim, embed_dim)

        self.fuse = FuseHead(embed_dim * num_seeds)

        if mixture_components > 1:
            self.prior_c = ZEncoder(z_dim, mixture_components)
            mdim = max(mixture_components, embed_dim)
            gaus = torch.randn(mdim, mdim)
            svd = torch.linalg.svd(gaus)
            orth = svd[0] @ svd[2]
            self.components_embedding = nn.Parameter(
                orth[:mixture_components, :embed_dim]
            )
        else:
            self.prior_c = None

        if flow_dim is None:
            flow_dim = embed_dim

        self.flow_decoder = self.build_flow(
            features=dim + 2,
            context=embed_dim,
            flow_type=flow_type,
            hidden_features=(flow_dim, flow_dim),
            transforms=flow_transforms,
        )

        self.label_enc = nn.Parameter(
            torch.randn(num_classes, num_seeds, embed_dim) * 0.01
        )
        self.label_enc_u = nn.Parameter(
            torch.randn(num_classes, num_seeds, embed_dim) * 0.01
        )

    def build_flow(self, features, context, flow_type, hidden_features, transforms):
        if flow_type == "naf":
            return zuko.flows.NAF(
                features,
                context,
                hidden_features=hidden_features,
                transforms=transforms,
            )
        elif flow_type == "maf":
            return zuko.flows.MAF(
                features,
                context,
                hidden_features=hidden_features,
                transforms=transforms,
            )
        elif flow_type == "unaf":
            return zuko.flows.UNAF(
                features,
                context,
                hidden_features=hidden_features,
                transforms=transforms,
            )
        else:
            raise ValueError(f"Unknown flow type: {flow_type}")

    def alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        return self.noise_schedule.alpha_bar(t)

    def x_enc(self, data, y):
        pos_embed = self.positional_encoding(data[..., :2])
        data = self.embed_data(torch.cat([data[..., 2:], pos_embed], dim=-1))
        return self.set_trans(data, y)

    def forward_u(self, y, z_t, t):
        t_emb = self.time_enc(t).unsqueeze(-2).repeat(1, z_t.size(1), 1)
        y_emb = self.label_enc_u[y]

        return self.fuse(y_emb, z_t, t_emb).view(*z_t.shape)

    def make_context(self, y, z_t, with_prior=False):
        comb = torch.cat([z_t, self.label_enc[y]], dim=-1)
        if with_prior:
            return self.z_encoder(comb.view(y.size(0), -1)), self.prior_c(
                comb.view(y.size(0), -1)
            )
        else:
            return self.z_encoder(comb.view(y.size(0), -1))

    def _loss_ce(self, x, y, z_t):
        context = self.make_context(y, z_t)
        return -self.flow_decoder(context).log_prob(x.moveaxis(0, 1)).sum(dim=0)

    def mixture_loss_ce(self, x, y, z_t):
        context, logits = self.make_context(y, z_t, with_prior=True)

        k = logits.shape[-1]  # number of components
        n, s, _ = x.shape

        # _cntxt = context.repeat(k, 1)
        _x = x.repeat(k, 1, 1)
        # _comp = self.components_embedding.view(k, 1, -1).repeat(1, n, 1).view(n * k, -1)

        c = context + self.components_embedding.view(
            k, 1, -1
        )  # torch.cat([_cntxt, _comp], dim=-1)
        log_prob = (
            self.flow_decoder(c.view(n * k, -1))
            .log_prob(_x.moveaxis(0, 1))
            .view(s, k, n)
            .mT
        )
        return -torch.logsumexp(log_prob + logits, dim=-1).sum(
            dim=0
        ) + s * torch.logsumexp(logits, dim=-1)

    def loss_ce(self, x, y, z_t):
        if self.prior_c:
            return self.mixture_loss_ce(x, y, z_t)
        else:
            return self._loss_ce(x, y, z_t)

    def sample(self, y, z_T, num_samples):
        if self.prior_c:
            context, logits = self.make_context(y, z_T, with_prior=True)
            n, k = logits.shape
            # have to do it sequential because of memory
            samples = []
            for _ in range(num_samples):
                comps = torch.distributions.Categorical(logits=logits).sample()
                # comps = self.components_embedding[comps]
                c = context + self.components_embedding[comps]
                samples.append(self.flow_decoder(c).sample())
            return torch.stack(samples, dim=1)
        else:
            context.model.make_context(y, z_T)
            return self.flow_decoder(context).sample((num_samples,)).moveaxis(0, 1)


# Training and Inference Logic
def compute_loss(
    model, x, y, device, η: float = 1.0, add_noise: bool = True, return_loss: int = -1
) -> float:
    B = x.size(0)
    u_x = model.x_enc(x, y)
    t = torch.rand(B, 1, device=device, requires_grad=True)
    αb = model.alpha_bar(t).unsqueeze(-1)
    snr = αb / (1 - αb)
    snr_p = torch.autograd.grad(snr.sum(), t, create_graph=True)[0]
    zt = αb.sqrt() * u_x + (1 - αb).sqrt() * torch.randn_like(u_x)
    pred_e = model.forward_u(y, zt, t)
    mse = F.mse_loss(pred_e, u_x, reduction="none").view(B, -1).sum(dim=1, keepdim=True)
    loss_sdm = 0.5 * η * (snr_p * mse).squeeze(-1)
    loss_kl = 0.5 * (u_x.pow(2).view(B, -1).sum(dim=1))
    t1 = torch.ones_like(t)
    αb1 = model.alpha_bar(t1).unsqueeze(-1)
    z1 = αb1.sqrt() * u_x + (1 - αb1).sqrt() * torch.randn_like(u_x)

    if add_noise:
        x = x + 0.05 * torch.randn_like(x)
    loss_ce = model.loss_ce(x, y, z1)

    if return_loss == -1:
        loss = loss_ce + loss_kl + loss_sdm
    elif return_loss == 0:
        loss = loss_ce
    elif return_loss == 1:
        loss = loss_kl
    elif return_loss == 2:
        loss = loss_sdm
    return loss


def train_step(model, x, y, device, η: float = 1.0) -> float:
    loss = compute_loss(model, x, y, device, η).mean()
    loss.backward()
    return loss.item()


@torch.no_grad()
def run_noprop_ct_inference_heun(
    model, y: torch.Tensor, num_samples: int, T_steps: int = 40
) -> torch.Tensor:
    model.eval()
    B = y.size(0)
    dt = 1.0 / T_steps
    z = torch.randn(B, *model.z_shape, device=y.device)
    for i in range(T_steps):
        t_n = torch.full((B, 1), i / T_steps, device=y.device)
        t_np1 = torch.full((B, 1), (i + 1) / T_steps, device=y.device)
        αn = model.alpha_bar(t_n).unsqueeze(-1)
        pred_n = model.forward_u(y, z, t_n)
        f_n = (pred_n - z) / (1 - αn)
        z_mid = z + dt * f_n
        αm = model.alpha_bar(t_np1).unsqueeze(-1)
        pred_mid = model.forward_u(y, z_mid, t_np1)
        f_mid = (pred_mid - z_mid) / (1 - αm)
        z = z + 0.5 * dt * (f_n + f_mid)

    z_T = model.forward_u(y, z, torch.ones_like(t_n))
    return model.sample(y, z_T, num_samples)


def patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    u = torch.nn.functional.unfold(x, patch_size, stride=patch_size)
    return u.transpose(1, 2)


def patched_image_to_pointcloud(
    x, patch_grid, num_samples=100, image_size=28, patch_size=4
):
    u = patchify(x, patch_size)
    pixel_args = torch.multinomial(u.mean(-1).abs(), num_samples, replacement=False)
    idxs = patch_grid[pixel_args]
    locs = 2 * (idxs + 0.5) / (image_size // patch_size) - 1
    pixel_val = torch.take_along_dim(2 * u - 1, pixel_args[..., None], dim=1)
    return torch.cat([locs, pixel_val], dim=-1)


def train_and_eval(
    time_emb_dim, embed_dim, batch_size, dataset, data_root, epoches, optim, flow
):
    print("start")
    # dataset-specific setup
    patch_size = 1
    if dataset == "mnist":
        ds_train = torchvision.datasets.MNIST(
            data_root,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    # transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        )
        ds_test = torchvision.datasets.MNIST(
            data_root,
            train=False,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    # transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        )
        num_classes = 10
        dim = patch_size**2
        image_size = 28
        dataset_size = 60_000

    elif dataset == "cifar10":
        # mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        ds_train = torchvision.datasets.CIFAR10(
            data_root,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    # transforms.Normalize(mean, std),
                ]
            ),
        )
        ds_test = torchvision.datasets.CIFAR10(
            data_root,
            train=False,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    # transforms.Normalize(mean, std),
                ]
            ),
        )
        num_classes = 10
        dim = 3 * (patch_size**2)
        image_size = 32
        dataset_size = 50_000

    elif dataset == "cifar100":
        # mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        ds_train = torchvision.datasets.CIFAR100(
            data_root,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomRotation(15),
                    transforms.RandomHorizontalFlip(),
                    # transforms.RandomAffine(10),
                    transforms.ToTensor(),
                    # transforms.Normalize(mean, std),
                ]
            ),
        )
        ds_test = torchvision.datasets.CIFAR100(
            data_root,
            train=False,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    # transforms.Normalize(mean, std),
                ]
            ),
        )
        num_classes = 100
        dim = 3 * (patch_size**2)
        image_size = 32
        dataset_size = 50_000

    else:
        raise ValueError(f"Unsupported dataset '{dataset}'")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- {dataset.upper()} ({num_classes} classes) using {device} ---")

    tr_loader = DataLoader(
        ds_train, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True
    )
    te_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=8)

    mixc = 16
    model = NoPropSetTransformerFlow(
        dim=dim,
        num_classes=num_classes,
        embed_dim=embed_dim,
        time_emb_dim=time_emb_dim,
        flow_type=flow,
        flow_transforms=2,
        flow_dim=32,
        mixture_components=mixc,
    ).to(device)

    if optim == "ivon":
        optimizer, scheduler = optim_sched(model, dataset_size, 1.0, 1e-5, epoches)
        train_samples = 1

    elif optim == "lamb":
        optimizer = torch_optimizer.Lamb(model.parameters(), lr=5e-3, weight_decay=1e-4)
        scheduler = None

    elif optim == "belief":
        optimizer = torch_optimizer.AdaBelief(
            model.parameters(), lr=1e-3, weight_decay=1e-4
        )
        scheduler = None

    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = None

    patch_grid_x, patch_grid_y = torch.meshgrid(
        torch.arange(image_size // patch_size),
        torch.arange(image_size // patch_size),
        indexing="ij",
    )
    patch_grid = torch.stack(
        [patch_grid_x.reshape(-1), patch_grid_y.reshape(-1)], dim=-1
    ).to(device)

    num_samples = 128
    η = 1.0

    print("train start")
    # training loop
    for ep in range(1, epoches + 1):
        t0, total_loss = time.time(), 0.0
        model.train()
        for x, y in tr_loader:
            x, y = x.to(device), y.to(device)
            x = patched_image_to_pointcloud(
                x,
                patch_grid,
                num_samples=num_samples,
                image_size=image_size,
                patch_size=patch_size,
            )
            if optim == "ivon":
                for _ in range(train_samples):
                    with optimizer.sampled_params(train=True):
                        optimizer.zero_grad()
                        total_loss += (
                            train_step(model, x, y, device, η)
                            * x.size(0)
                            / train_samples
                        )
                optimizer.step()
            else:
                optimizer.zero_grad()
                total_loss += train_step(model, x, y, device, η) * x.size(0)
                optimizer.step()

        scheduler.step() if scheduler is not None else None

        if ep % 5 == 0:
            avg_loss = total_loss / len(ds_train)
            print()
            print(
                f" Epoch {ep:03d} loss {avg_loss:8.4f} | train {time.time()-t0:3.1f}s",
                end="",
            )

        if ep % 10 == 0:
            model.eval()
            tot = acc = 0.0
            eval_t0 = time.time()
            for x, y in te_loader:
                x, y = x.to(device), y.to(device)
                x = patched_image_to_pointcloud(
                    x,
                    patch_grid,
                    num_samples=num_samples,
                    image_size=image_size,
                    patch_size=patch_size,
                )
                preds = run_noprop_ct_inference_heun(model, y, 256, T_steps=20)
                tot += y.size(0)
                acc += evaluate_acc(
                    model, x, y, num_classes, device, num_samples=10, η=η
                )

            fig, axes = plt.subplots(10, 5, figsize=(6, 8), sharex=True, sharey=True)

            for j in range(5):
                for i in range(10):
                    x = preds[y == i][j].cpu().numpy()
                    axes[i, j].hist2d(
                        x[:, 1], -x[:, 0], bins=28, range=((-1, 1), (-1, 1))
                    )
                    axes[i, j].set_yticks([])
                    axes[i, j].set_xticks([])

            fig.tight_layout()
            fig.savefig(f"mnist_gen_{ep}_{mixc}.png", dpi=100)

            print(
                f" | ACC {100 * acc/tot:4.2f}% | time {time.time()-eval_t0:3.1f}s",
                end="",
            )

    # cleanup
    del model, optimizer, ds_train, ds_test, tr_loader, te_loader
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default="mnist", choices=["mnist", "cifar10", "cifar100"]
    )
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--time-emb-dim", type=int, default=64)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--flow", type=str, default="maf", choices=["maf"])
    args = parser.parse_args()

    print("argparse done", args)
    train_and_eval(
        time_emb_dim=args.time_emb_dim,
        embed_dim=args.embed_dim,
        batch_size=args.batch_size,
        dataset=args.dataset,
        data_root=args.data_root,
        epoches=args.epochs,
        optim=args.optimizer,
        flow=args.flow,
    )
