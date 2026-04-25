import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.generated import VitConfig


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_value: float) -> None:
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gamma


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        # (B, N, C) -> (B, N, 3*C) -> (B, N, 3, num_heads, head_dim) -> (3, B, num_heads, N, head_dim)
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        # q, k, v: (B, num_heads, N, head_dim)
        q, k, v = qkv.unbind(0)
        out = F.scaled_dot_product_attention(q, k, v)
        # (B, num_heads, N, head_dim) -> (B, N, num_heads * head_dim) -> (B, N, C)
        return self.proj(out.transpose(1, 2).reshape(B, N, C))


class EncoderBlock(nn.Module):
    def __init__(self, cfg: VitConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.embed_dim)
        self.attn = Attention(cfg.embed_dim, cfg.num_heads)
        self.ls1 = LayerScale(cfg.embed_dim, cfg.ls_init_value)
        self.norm2 = nn.LayerNorm(cfg.embed_dim)
        self.mlp = MLP(cfg.embed_dim, int(cfg.embed_dim * cfg.mlp_ratio))
        self.ls2 = LayerScale(cfg.embed_dim, cfg.ls_init_value)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size: int, in_channels: int, embed_dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class VisionTransformer(nn.Module):
    def __init__(self, cfg: VitConfig) -> None:
        super().__init__()
        self.embed_dim = cfg.embed_dim
        num_patches = (cfg.img_size // cfg.patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        self.register_tokens = nn.Parameter(
            torch.zeros(1, cfg.num_register_tokens, cfg.embed_dim)
        )
        # While CLS token has positional embedding, register tokens do not have positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, cfg.embed_dim))
        self.patch_embed = PatchEmbedding(
            cfg.patch_size, cfg.in_channels, cfg.embed_dim
        )
        self.blocks = nn.ModuleList([EncoderBlock(cfg) for _ in range(cfg.depth)])
        self.norm = nn.LayerNorm(cfg.embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        B = x.shape[0]

        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls, x), dim=1) + self.pos_embed

        cls_with_pos = x[:, :1]
        patches_with_pos = x[:, 1:]
        registers = self.register_tokens.expand(B, -1, -1)
        x = torch.cat((cls_with_pos, registers, patches_with_pos), dim=1)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        # Remove register tokens before returning
        num_register_tokens = self.register_tokens.shape[1]
        return torch.cat((x[:, :1], x[:, 1 + num_register_tokens :]), dim=1)
