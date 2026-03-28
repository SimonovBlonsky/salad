import torch
import torch.nn as nn
import pytest

from ablations.models.dinov2_intermediate import DINOv2IntermediateEncoder


class DummyBlock(nn.Module):
    def __init__(self, layer_index, channels):
        super().__init__()
        self.layer_index = layer_index
        self.weight = nn.Parameter(torch.tensor(float(layer_index + 1)))
        self.channels = channels

    def forward(self, x):
        x = x.clone()
        x[:, 0] = x[:, 0] + self.weight
        x[:, 1:] = x[:, 1:] + self.weight
        return x


class DummyNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.5))
        self.channels = channels

    def forward(self, x):
        return x + self.bias


class DummyDinov2Backbone(nn.Module):
    def __init__(self, num_blocks=3, channels=768, patch_size=14):
        super().__init__()
        self.patch_size = patch_size
        self.channels = channels
        self.blocks = nn.ModuleList([DummyBlock(i, channels) for i in range(num_blocks)])
        self.norm = DummyNorm(channels)
        self.prepare_tokens_calls = 0
        self.train_modes = []

    def prepare_tokens_with_masks(self, x):
        self.prepare_tokens_calls += 1
        batch, _, height, width = x.shape
        hp = height // self.patch_size
        wp = width // self.patch_size
        num_tokens = hp * wp
        tokens = torch.arange(
            batch * (num_tokens + 1) * self.channels, dtype=x.dtype, device=x.device
        ).reshape(batch, num_tokens + 1, self.channels)
        return tokens

    def train(self, mode=True):
        self.train_modes.append(mode)
        return super().train(mode)


def make_encoder(**kwargs):
    backbone_kwargs = kwargs.pop("backbone_kwargs", {})
    backbone = DummyDinov2Backbone(**backbone_kwargs)

    def hub_loader(repo, model_name):
        assert repo == "facebookresearch/dinov2"
        assert model_name == kwargs.get("model_name", "dinov2_vitb14")
        return backbone

    encoder = DINOv2IntermediateEncoder(
        hub_loader=hub_loader,
        **kwargs,
    )
    return encoder, backbone


def test_output_layer_one_returns_intermediate_output():
    encoder, backbone = make_encoder(output_layer=1)

    x = torch.zeros(2, 3, 28, 28)
    features = encoder(x)

    assert encoder.num_channels == 768
    assert backbone.prepare_tokens_calls == 1
    assert set(features) == {"patch_tokens", "feature_map", "global_token", "spatial_shape"}
    assert features["patch_tokens"].shape == (2, 4, 768)
    assert features["feature_map"].shape == (2, 768, 2, 2)
    assert features["global_token"].shape == (2, 768)
    assert features["spatial_shape"] == (2, 2)
    expected = torch.arange(2 * 5 * 768, dtype=torch.float32).reshape(2, 5, 768) + 3.5
    assert torch.equal(features["global_token"], expected[:, 0])
    assert torch.equal(features["patch_tokens"], expected[:, 1:])
    assert torch.equal(
        features["feature_map"],
        expected[:, 1:].reshape(2, 2, 2, 768).permute(0, 3, 1, 2),
    )
    assert torch.isfinite(features["patch_tokens"]).all()
    assert torch.isfinite(features["feature_map"]).all()
    assert torch.isfinite(features["global_token"]).all()


def test_output_layer_minus_one_returns_final_output():
    encoder, _ = make_encoder(output_layer=-1)

    x = torch.zeros(1, 3, 28, 28)
    features = encoder(x)

    assert features["patch_tokens"].shape == (1, 4, 768)
    assert features["feature_map"].shape == (1, 768, 2, 2)
    assert features["global_token"].shape == (1, 768)
    assert features["spatial_shape"] == (2, 2)
    expected = torch.arange(1 * 5 * 768, dtype=torch.float32).reshape(1, 5, 768) + 6.5
    assert torch.equal(features["global_token"], expected[:, 0])
    assert torch.equal(features["patch_tokens"], expected[:, 1:])


def test_freeze_backbone_sets_all_parameters_non_trainable_and_stays_eval():
    encoder, backbone = make_encoder(freeze_backbone=True)

    assert all(not parameter.requires_grad for parameter in encoder.parameters())

    encoder.train(True)

    assert not encoder.training
    assert not backbone.training
    assert backbone.train_modes[-1] is False


def test_partial_freeze_unfreezes_only_last_blocks_and_norm_when_enabled():
    encoder, backbone = make_encoder(
        num_trainable_blocks=2,
        norm_layer=True,
        backbone_kwargs={"num_blocks": 4},
    )

    block_requires_grad = [
        [parameter.requires_grad for parameter in block.parameters()]
        for block in backbone.blocks
    ]

    assert block_requires_grad[0] == [False]
    assert block_requires_grad[1] == [False]
    assert block_requires_grad[2] == [True]
    assert block_requires_grad[3] == [True]
    assert backbone.norm.bias.requires_grad


def test_partial_freeze_keeps_norm_frozen_when_disabled():
    encoder, backbone = make_encoder(
        num_trainable_blocks=2,
        norm_layer=False,
        backbone_kwargs={"num_blocks": 4},
    )

    block_requires_grad = [
        [parameter.requires_grad for parameter in block.parameters()]
        for block in backbone.blocks
    ]

    assert block_requires_grad[0] == [False]
    assert block_requires_grad[1] == [False]
    assert block_requires_grad[2] == [True]
    assert block_requires_grad[3] == [True]
    assert not backbone.norm.bias.requires_grad

    encoder.train(True)

    assert not backbone.norm.training


def test_partial_freeze_keeps_frozen_prefix_eval_and_trainable_suffix_train():
    encoder, backbone = make_encoder(
        num_trainable_blocks=2,
        norm_layer=True,
        backbone_kwargs={"num_blocks": 4},
    )

    encoder.train(True)

    assert encoder.training
    assert not backbone.blocks[0].training
    assert not backbone.blocks[1].training
    assert backbone.blocks[2].training
    assert backbone.blocks[3].training
    assert backbone.norm.training


def test_non_positive_trainable_blocks_freezes_backbone_and_stays_eval():
    encoder, backbone = make_encoder(num_trainable_blocks=0)

    assert all(not parameter.requires_grad for parameter in encoder.parameters())

    encoder.train(True)

    assert not encoder.training
    assert not backbone.training
    assert backbone.train_modes[-1] is False


def test_intermediate_output_layer_routes_gradients_to_selected_trainable_suffix():
    encoder, backbone = make_encoder(
        output_layer=1,
        num_trainable_blocks=1,
        norm_layer=True,
        backbone_kwargs={"num_blocks": 3},
    )

    encoder.train(True)

    features = encoder(torch.zeros(1, 3, 28, 28))
    features["patch_tokens"].sum().backward()

    assert backbone.blocks[0].weight.grad is None
    assert backbone.blocks[1].weight.grad is not None
    assert backbone.blocks[2].weight.grad is None
    assert backbone.norm.bias.grad is not None


def test_invalid_output_layer_raises_value_error():
    with pytest.raises(ValueError, match="output_layer"):
        make_encoder(output_layer=3)

    with pytest.raises(ValueError, match="output_layer"):
        make_encoder(output_layer=-2)


def test_invalid_patch_token_reshape_raises_value_error():
    class MismatchedTokensBackbone(DummyDinov2Backbone):
        def prepare_tokens_with_masks(self, x):
            batch, _, _, _ = x.shape
            return torch.zeros(batch, 6, self.channels, dtype=x.dtype, device=x.device)

    backbone = MismatchedTokensBackbone()

    def hub_loader(repo, model_name):
        return backbone

    encoder = DINOv2IntermediateEncoder(hub_loader=hub_loader)

    with pytest.raises(ValueError, match="patch tokens"):
        encoder(torch.zeros(1, 3, 30, 28))


def test_invalid_model_name_raises_value_error():
    with pytest.raises(ValueError, match="Unknown model name"):
        DINOv2IntermediateEncoder(model_name="not_a_model", hub_loader=lambda *_: None)


def test_non_finite_returned_feature_raises_value_error():
    class NonFiniteBlock(DummyBlock):
        def forward(self, x):
            x = super().forward(x)
            x[:, 0, 0] = float("nan")
            return x

    class NonFiniteBackbone(DummyDinov2Backbone):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([DummyBlock(0, self.channels), NonFiniteBlock(1, self.channels)])

    encoder = DINOv2IntermediateEncoder(hub_loader=lambda *_: NonFiniteBackbone(), output_layer=-1)

    with pytest.raises(ValueError, match="non-finite"):
        encoder(torch.zeros(1, 3, 28, 28))
