import pytest
import torch

from src.models.vit import VisionTransformer, VitConfig
from src.utils.checkpoint import load_weights


@pytest.fixture
def vit_config() -> VitConfig:
    return VitConfig(
        img_size=518,
        patch_size=14,
        in_channels=3,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        ls_init_value=1e-6,
        num_register_tokens=4,
    )


@pytest.fixture
def model(vit_config: VitConfig) -> VisionTransformer:
    return VisionTransformer(vit_config)


def test_load_dino_ckpt(model: VisionTransformer) -> None:
    ckpt = torch.load(
        "checkpoints/dinov2_vits14_reg4_pretrain.pth",
        weights_only=True,
    )

    before = {k: v.clone() for k, v in model.state_dict().items()}
    load_weights(model, ckpt)
    after = model.state_dict()

    not_updated = [k for k in before if torch.allclose(before[k], after[k])]
    assert not not_updated


def test_vit_forward_runs(model: VisionTransformer) -> None:
    x = torch.randn(1, 3, 518, 518)
    with torch.inference_mode():
        model(x)
