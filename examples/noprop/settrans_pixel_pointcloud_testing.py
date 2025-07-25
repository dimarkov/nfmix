from functools import wraps
import argparse
import time
import gc

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms
from einops import rearrange, repeat

import torch_optimizer
import ivon
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

# helpers


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def cache_fn(f):
    cache = None

    @wraps(f)
    def cached_fn(*args, _cache=True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache

    return cached_fn


# ----------------------------------------------------------------------------
# positional encoding
# ----------------------------------------------------------------------------


class GaussianPositionalEncoding(torch.nn.Module):
    def __init__(self, channels, num_freqs=16, sigma=1.0):
        super().__init__()
        self.num_freqs = num_freqs

        self.b = torch.nn.Parameter(torch.randn((num_freqs, 2)) * sigma)
        # self.register_buffer('b', b)
        fourier_dim = 2 * num_freqs

        self.proj = torch.nn.Linear(fourier_dim, channels)

    def forward(self, coords):
        coords_freqs = (2 * torch.pi * coords) @ self.b.T
        features = torch.cat([torch.sin(coords_freqs), torch.cos(coords_freqs)], dim=-1)

        return self.proj(features)


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
    def __init__(self, dim, heads=8, dim_head=64, num_inducing_points=16):
        super().__init__()
        self.inducing_points = nn.Parameter(torch.randn(1, num_inducing_points, dim))
        self.mab1 = MAB(dim, heads=heads, dim_head=dim_head)
        self.mab2 = MAB(dim, heads=heads, dim_head=dim_head)

    def forward(self, x):
        h = self.mab1(self.inducing_points.repeat(x.size(0), 1, 1), x)
        return self.mab2(x, h)


class PMA(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, num_seeds=1):
        super().__init__()
        self.seeds = nn.Parameter(torch.randn(1, num_seeds, dim))
        self.mab = MAB(dim, heads=heads, dim_head=dim_head)
        self.ff = FeedForward(dim)

    def forward(self, x):
        x = self.ff(x)
        return self.mab(self.seeds.repeat(x.size(0), 1, 1), x)


class SetTransformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        pos_dim,
        embed_dim,
        depth_enc,
        depth_dec,
        heads,
        dim_head,
        num_inducing_points,
        dim_out,
        num_seeds=1,
        use_pos_enc=True,
    ):
        super().__init__()
        self.embedding = nn.Linear(dim + pos_dim, embed_dim)
        self.encoder = nn.Sequential(
            *[
                ISAB(
                    embed_dim,
                    heads=heads,
                    dim_head=dim_head,
                    num_inducing_points=num_inducing_points,
                )
                for _ in range(depth_enc)
            ]
        )
        self.decoder = nn.Sequential(
            PMA(embed_dim, heads=heads, dim_head=dim_head, num_seeds=num_seeds),
            *[SAB(embed_dim, heads=heads, dim_head=dim_head) for _ in range(depth_dec)],
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim),
        )

        self.fc = nn.Linear(num_seeds * embed_dim, dim_out)

        self.pos_enc = GaussianPositionalEncoding(pos_dim) if use_pos_enc else None

    def forward(self, x):
        if self.pos_enc is not None:
            pos_embed = self.pos_enc(x[..., :2])
            features = x[..., 2:]
            x = self.embedding(torch.cat([features, pos_embed], dim=-1))
        else:
            x = self.embedding(x)

        for enc in self.encoder:
            x = enc(x)

        x = self.decoder[0](x)
        for dec in self.decoder[1:]:
            x = dec(x)

        return self.fc(x.view(x.size(0), -1))


def optim_sched(model, datasize, hess_init, weight_decay, num_epochs):
    optimizer = ivon.IVON(
        model.parameters(),
        lr=1e-1,
        weight_decay=weight_decay,
        hess_init=hess_init,
        ess=datasize,
    )
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_epochs=num_epochs // 5,
        max_epochs=num_epochs,
        warmup_start_lr=1e-2,
        eta_min=1e-3,
    )
    return optimizer, scheduler


def evaluate_acc(model, x, y):
    logits = model(x)
    num_correct = (logits.argmax(dim=-1) == y).sum().item()
    return num_correct


# Training and Inference Logic
def compute_loss(model, x, y) -> float:
    logits = model(x)
    return F.cross_entropy(logits, y)


def train_step(model, x, y, device, η: float = 1.0) -> float:
    loss = compute_loss(model, x, y).mean()
    loss.backward()
    return loss.item()


def patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    u = torch.nn.functional.unfold(x, patch_size, stride=patch_size)
    return u.transpose(1, 2)


def patched_image_to_pointcloud(
    x, patch_grid, num_samples=100, image_size=28, patch_size=4, use_pos_enc=True
):
    u = patchify(x, patch_size)
    pixel_args = torch.multinomial(u.mean(-1).abs(), num_samples, replacement=False)
    idxs = patch_grid[pixel_args]
    locs = 2 * (idxs + 0.5) / (image_size // patch_size) - 1
    pixel_val = torch.take_along_dim(2 * u - 1, pixel_args[..., None], dim=1)

    if use_pos_enc:
        return torch.cat([locs, pixel_val], dim=-1)
    else:
        return pixel_val


def train_and_eval(
    embed_dim,
    batch_size,
    dataset,
    data_root,
    epoches,
    optim,
):
    print("start")
    # dataset-specific setup
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
    dim = 1
    image_size = 28
    dataset_size = 60_000
    pos_encoding = True
    num_samples = 64

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- {dataset.upper()} ({num_classes} classes) using {device} ---")

    tr_loader = DataLoader(
        ds_train, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True
    )
    te_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=8)

    model = SetTransformer(
        dim=dim,
        pos_dim=16,
        embed_dim=embed_dim,
        dim_out=num_classes,
        depth_enc=2,
        depth_dec=1,
        heads=8,
        dim_head=64,
        num_inducing_points=4,
        num_seeds=2,
        use_pos_enc=pos_encoding,
    ).to(device)

    if optim == "ivon":
        optimizer, scheduler = optim_sched(model, dataset_size, 0.1, 1e-5, epoches)
        train_samples = 1

    elif optim == "lamb":
        optimizer = torch_optimizer.Lamb(model.parameters(), lr=1e-3, weight_decay=1e-3)
        scheduler = None

    elif optim == "belief":
        optimizer = torch_optimizer.AdaBelief(
            model.parameters(), lr=1e-3, weight_decay=1e-3
        )
        scheduler = None

    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        scheduler = None

    patch_size = 1
    patch_grid_x, patch_grid_y = torch.meshgrid(
        torch.arange(image_size // patch_size),
        torch.arange(image_size // patch_size),
        indexing="ij",
    )
    patch_grid = torch.stack(
        [patch_grid_x.reshape(-1), patch_grid_y.reshape(-1)], dim=-1
    ).to(device)

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
                use_pos_enc=pos_encoding,
            )
            if optim == "ivon":
                for _ in range(train_samples):
                    with optimizer.sampled_params(train=True):
                        optimizer.zero_grad()
                        total_loss += (
                            train_step(model, x, y, device) * x.size(0) / train_samples
                        )
                optimizer.step()
            else:
                optimizer.zero_grad()
                total_loss += train_step(model, x, y, device) * x.size(0)
                optimizer.step()

        scheduler.step() if scheduler is not None else None
        avg_loss = total_loss / len(ds_train)
        print(
            f" Epoch {ep:03d} loss {avg_loss:8.4f} | train {time.time()-t0:3.1f}s",
            end="",
        )

        if ep % 5 == 0:
            model.eval()
            tot = acc = 0
            eval_t0 = time.time()
            for x, y in te_loader:
                x, y = x.to(device), y.to(device)
                x = patched_image_to_pointcloud(
                    x,
                    patch_grid,
                    num_samples=num_samples,
                    image_size=image_size,
                    patch_size=patch_size,
                    use_pos_enc=pos_encoding,
                )
                tot += y.size(0)
                acc += evaluate_acc(model, x, y)

            print(
                f" | ACC {100 * acc/tot:4.2f}% | time {time.time()-eval_t0:3.1f}s",
                end="",
            )
        print()

    # cleanup
    del model, optimizer, ds_train, ds_test, tr_loader, te_loader
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="mnist", choices=["mnist"])
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--optimizer", type=str, default="adamw")
    args = parser.parse_args()

    print("argparse done", args)
    train_and_eval(
        embed_dim=args.embed_dim,
        batch_size=args.batch_size,
        dataset=args.dataset,
        data_root=args.data_root,
        epoches=args.epochs,
        optim=args.optimizer,
    )
