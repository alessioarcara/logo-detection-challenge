import pytest
import torch

from src.models.neck import DPTNeck, FusionBlock, ReassembleBlock

GRID = 37  # 518 / 14
DIM = 384
FEAT = 256
B = 2


@pytest.fixture()
def tokens() -> torch.Tensor:
    return torch.randn(B, GRID * GRID, DIM)


@pytest.fixture()
def hidden_states() -> list[torch.Tensor]:
    return [torch.randn(B, GRID * GRID, DIM) for _ in range(12)]


@pytest.fixture()
def neck() -> DPTNeck:
    return DPTNeck(embed_dim=DIM, features=FEAT)


@pytest.mark.parametrize(
    "scale, expected_h", [(1.0, 37), (2.0, 74), (4.0, 148), (0.5, 19)]
)
def test_reassemble_output_shape(
    tokens: torch.Tensor, scale: float, expected_h: int
) -> None:
    out = ReassembleBlock(DIM, FEAT, scale=scale)(tokens, GRID)
    assert out.shape == (B, FEAT, expected_h, expected_h)


def test_fusion_no_skip() -> None:
    out = FusionBlock(FEAT)(torch.randn(B, FEAT, 19, 19))
    assert out.shape == (B, FEAT, 38, 38)


def test_fusion_with_skip() -> None:
    x = torch.randn(B, FEAT, 38, 38)
    skip = torch.randn(B, FEAT, 37, 37)
    out = FusionBlock(FEAT)(x, skip)
    assert out.shape == (B, FEAT, 74, 74)


@pytest.mark.parametrize("grid, expected_h", [(37, 296), (24, 192)])
def test_neck_output_shape(neck: DPTNeck, grid: int, expected_h: int) -> None:
    hs = [torch.randn(B, grid * grid, DIM) for _ in range(12)]
    out = neck(hs, grid)
    assert out.shape == (B, FEAT, expected_h, expected_h)
